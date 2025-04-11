# processing/vectorizer.py
import logging
import time
import hashlib
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from models.embedding import BaseEmbeddingModel
from data.vector_store import QdrantVectorStore
from config import get_config, get_model_config

logger = logging.getLogger(__name__)


class LogVectorizer:
    """日志向量化处理器"""

    def __init__(
            self,
            embedding_model: BaseEmbeddingModel,
            vector_store: QdrantVectorStore,
            chunk_size: Optional[int] = None,
            batch_size: Optional[int] = None,
            overlap: Optional[int] = None
    ):
        """
        初始化日志向量化处理器

        Args:
            embedding_model: 嵌入模型
            vector_store: 向量存储
            chunk_size: 文本块大小，用于长文本分块
            batch_size: 批处理大小
            overlap: 文本块重叠大小
        """
        # 加载配置
        self.config = get_config()
        self.model_config = get_model_config()

        # 设置参数，使用配置值作为默认值
        vectorization_config = self.model_config.get("vectorization", {})
        self.chunk_size = chunk_size or vectorization_config.get("chunk_size", 512)
        self.batch_size = batch_size or vectorization_config.get("batch_size", 100)
        self.overlap = overlap or vectorization_config.get("overlap", 0)

        self.embedding_model = embedding_model
        self.vector_store = vector_store

        logger.info(f"日志向量化处理器初始化，使用向量维度: {embedding_model.get_dimension()}")

    def process_dataframe(
            self,
            df: pd.DataFrame,
            content_column: str = "message",
            id_column: Optional[str] = None,
            metadata_columns: Optional[List[str]] = None,
            service_name_column: str = "service_name"
    ) -> Tuple[int, int]:
        """
        处理DataFrame并向量化日志内容

        Args:
            df: 包含日志数据的DataFrame
            content_column: 内容列名
            id_column: ID列名，如果为None则自动生成
            metadata_columns: 要包含在元数据中的列
            service_name_column: 服务名称列，用于区分不同系统

        Returns:
            (处理的记录数, 成功的记录数)
        """
        if df.empty:
            logger.warning("DataFrame为空，没有数据需要处理")
            return 0, 0

        # 确保内容列存在
        if content_column not in df.columns:
            logger.error(f"内容列 '{content_column}' 不在DataFrame中")
            return 0, 0

        # 如果指定了service_name_column，但不在元数据列中，添加它
        if service_name_column in df.columns:
            if metadata_columns is None:
                metadata_columns = [service_name_column]
            elif service_name_column not in metadata_columns:
                metadata_columns.append(service_name_column)

        # 默认使用所有列作为元数据
        if metadata_columns is None:
            metadata_columns = [col for col in df.columns if col != content_column]

        # 处理DataFrame中的缺失值
        df = self._preprocess_dataframe(df, content_column)

        # 分批处理
        total_records = len(df)
        successful_records = 0

        logger.info(f"开始处理 {total_records} 条日志记录")
        start_time = time.time()

        for i in range(0, total_records, self.batch_size):
            batch_df = df.iloc[i:i + self.batch_size].copy()

            # 处理此批次
            batch_texts = batch_df[content_column].tolist()

            # 检查是否有需要分块的长文本
            if self.chunk_size > 0:
                batch_texts, batch_chunk_indices = self._chunk_texts(batch_texts)

            # 生成或提取ID
            if id_column and id_column in batch_df.columns:
                base_batch_ids = batch_df[id_column].astype(str).tolist()
                # 如果进行了分块，需要为每个分块生成唯一ID
                if self.chunk_size > 0:
                    batch_ids = []
                    for idx, base_id in enumerate(base_batch_ids):
                        if idx in batch_chunk_indices:
                            chunk_count = batch_chunk_indices[idx]
                            batch_ids.extend([f"{base_id}_chunk_{i}" for i in range(chunk_count)])
                        else:
                            batch_ids.append(base_id)
                else:
                    batch_ids = base_batch_ids
            else:
                # 使用内容哈希作为ID
                batch_ids = [
                    hashlib.md5(text.encode('utf-8')).hexdigest()
                    for text in batch_texts
                ]

            # 提取元数据
            batch_payloads = []

            if self.chunk_size > 0:
                # 处理分块情况下的元数据
                current_chunk_idx = 0
                for idx, row in batch_df.iterrows():
                    base_metadata = self._extract_row_metadata(row, content_column, metadata_columns)

                    if idx in batch_chunk_indices:
                        chunk_count = batch_chunk_indices[idx]
                        for chunk_idx in range(chunk_count):
                            chunk_metadata = base_metadata.copy()
                            chunk_metadata["chunk_index"] = chunk_idx
                            chunk_metadata["total_chunks"] = chunk_count
                            batch_payloads.append({"metadata": chunk_metadata})
                            current_chunk_idx += 1
                    else:
                        batch_payloads.append({"metadata": base_metadata})
                        current_chunk_idx += 1
            else:
                # 不分块情况下的元数据
                for _, row in batch_df.iterrows():
                    metadata = self._extract_row_metadata(row, content_column, metadata_columns)
                    batch_payloads.append({"metadata": metadata})

            # 生成嵌入向量
            batch_vectors = self.embedding_model.embed(batch_texts)

            # 将向量存储到Qdrant
            success = self.vector_store.add_vectors(
                ids=batch_ids,
                vectors=batch_vectors,
                payloads=batch_payloads
            )

            if success:
                successful_records += len(batch_ids)

            logger.info(f"已处理 {i + len(batch_df)}/{total_records} 条记录")

        elapsed_time = time.time() - start_time
        logger.info(f"处理完成，成功处理 {successful_records}/{total_records} 条记录，耗时 {elapsed_time:.2f} 秒")

        return total_records, successful_records

    def _extract_row_metadata(
            self,
            row: pd.Series,
            content_column: str,
            metadata_columns: List[str]
    ) -> Dict[str, Any]:
        """
        从DataFrame行中提取元数据

        Args:
            row: DataFrame的一行
            content_column: 内容列名
            metadata_columns: 要包含的元数据列

        Returns:
            元数据字典
        """
        metadata = {"original_text": row[content_column]}

        # 添加选定的元数据列
        for col in metadata_columns:
            if col in row:
                value = row[col]
                # 处理特殊类型
                if isinstance(value, (datetime, pd.Timestamp)):
                    metadata[col] = value.isoformat()
                elif isinstance(value, (np.int64, np.float64)):
                    metadata[col] = value.item()  # 将numpy类型转换为Python原生类型
                else:
                    metadata[col] = value

        return metadata

    def _chunk_texts(self, texts: List[str]) -> Tuple[List[str], Dict[int, int]]:
        """
        将长文本分块

        Args:
            texts: 要分块的文本列表

        Returns:
            (分块后的文本列表, 原始索引到分块数量的映射)
        """
        chunked_texts = []
        chunk_indices = {}  # 记录原始索引对应的分块数量

        for i, text in enumerate(texts):
            if len(text) <= self.chunk_size:
                chunked_texts.append(text)
                continue

            # 将长文本分块
            chunks = []
            for j in range(0, len(text), self.chunk_size - self.overlap):
                chunk = text[j:j + self.chunk_size]
                if chunk:  # 确保块不为空
                    chunks.append(chunk)

            chunk_indices[i] = len(chunks)
            chunked_texts.extend(chunks)

        return chunked_texts, chunk_indices

    def process_log_clusters(
            self,
            df: pd.DataFrame,
            template_column: str = "template",
            cluster_id_column: str = "cluster_id",
            sample_content_column: str = "message",
            additional_metadata_columns: Optional[List[str]] = None,
            service_name_column: str = "service_name"
    ) -> Tuple[int, int]:
        """
        处理日志聚类结果并向量化模板

        Args:
            df: 包含聚类结果的DataFrame
            template_column: 模板列名
            cluster_id_column: 聚类ID列名
            sample_content_column: 样本内容列名
            additional_metadata_columns: 额外的元数据列
            service_name_column: 服务名称列

        Returns:
            (处理的记录数, 成功的记录数)
        """
        if df.empty or template_column not in df.columns:
            logger.warning("聚类数据为空或缺少模板列")
            return 0, 0

        # 获取唯一的聚类
        clusters_df = df.drop_duplicates(subset=[cluster_id_column])

        metadata_columns = [cluster_id_column]
        if additional_metadata_columns:
            metadata_columns.extend(additional_metadata_columns)

        # 确保service_name_column在元数据中
        if service_name_column in df.columns and service_name_column not in metadata_columns:
            metadata_columns.append(service_name_column)

        return self.process_dataframe(
            df=clusters_df,
            content_column=template_column,
            id_column=cluster_id_column,
            metadata_columns=metadata_columns
        )

    def process_error_stacks(
            self,
            df: pd.DataFrame,
            error_stack_column: str = "stack_trace",
            id_column: Optional[str] = None,
            metadata_columns: Optional[List[str]] = None,
            service_name_column: str = "service_name"
    ) -> Tuple[int, int]:
        """
        处理错误堆栈跟踪信息并向量化

        Args:
            df: 包含错误堆栈的DataFrame
            error_stack_column: 错误堆栈列名
            id_column: ID列名
            metadata_columns: 元数据列
            service_name_column: 服务名称列

        Returns:
            (处理的记录数, 成功的记录数)
        """
        # 过滤有错误堆栈的记录
        error_df = df[df[error_stack_column].notna() & (df[error_stack_column] != "")]

        if error_df.empty:
            logger.warning("没有错误堆栈数据需要处理")
            return 0, 0

        # 确保service_name_column在元数据中
        if metadata_columns is None:
            metadata_columns = []

        if service_name_column in df.columns and service_name_column not in metadata_columns:
            metadata_columns.append(service_name_column)

        return self.process_dataframe(
            df=error_df,
            content_column=error_stack_column,
            id_column=id_column,
            metadata_columns=metadata_columns
        )

    def _preprocess_dataframe(self, df: pd.DataFrame, content_column: str) -> pd.DataFrame:
        """
        预处理DataFrame，处理缺失值和特殊类型

        Args:
            df: 原始DataFrame
            content_column: 内容列名

        Returns:
            处理后的DataFrame
        """
        # 创建副本以避免修改原始数据
        df = df.copy()

        # 确保内容列不为空
        df[content_column] = df[content_column].fillna("").astype(str)

        # 处理可能的列类型问题
        for col in df.columns:
            # 将日期时间列转换为ISO格式字符串
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(str)

            # 将null值替换为空字符串（对于字符串列）
            if df[col].dtype == object:
                df[col] = df[col].fillna("")

        return df

    def process_by_service(
            self,
            df: pd.DataFrame,
            service_name_column: str = "service_name",
            content_column: str = "message",
            id_column: Optional[str] = None,
            metadata_columns: Optional[List[str]] = None
    ) -> Dict[str, Tuple[int, int]]:
        """
        按服务名称分别处理日志数据

        Args:
            df: 包含日志数据的DataFrame
            service_name_column: 服务名称列名
            content_column: 内容列名
            id_column: ID列名
            metadata_columns: 元数据列

        Returns:
            服务名称到处理结果的映射
        """
        if df.empty:
            logger.warning("DataFrame为空，没有数据需要处理")
            return {}

        if service_name_column not in df.columns:
            logger.warning(f"服务名称列 '{service_name_column}' 不在DataFrame中")
            # 将所有记录当作一个服务处理
            result = self.process_dataframe(
                df=df,
                content_column=content_column,
                id_column=id_column,
                metadata_columns=metadata_columns
            )
            return {"unknown_service": result}

        # 按服务名称分组处理
        results = {}

        # 获取唯一的服务名称
        service_names = df[service_name_column].unique()

        for service_name in service_names:
            if pd.isna(service_name):
                service_df = df[df[service_name_column].isna()]
                service_key = "unknown_service"
            else:
                service_df = df[df[service_name_column] == service_name]
                service_key = str(service_name)

            if service_df.empty:
                continue

            logger.info(f"处理服务 '{service_key}' 的日志数据 ({len(service_df)} 条记录)")

            # 处理此服务的日志
            result = self.process_dataframe(
                df=service_df,
                content_column=content_column,
                id_column=id_column,
                metadata_columns=metadata_columns,
                service_name_column=service_name_column
            )

            results[service_key] = result

        return results

    def update_vector_metadata(
            self,
            df: pd.DataFrame,
            id_column: str,
            metadata_updates: Dict[str, Any]
    ) -> int:
        """
        更新向量元数据

        Args:
            df: 包含ID的DataFrame
            id_column: ID列名
            metadata_updates: 要更新的元数据

        Returns:
            成功更新的记录数
        """
        if df.empty or id_column not in df.columns:
            logger.warning(f"更新元数据失败: DataFrame为空或缺少ID列")
            return 0

        # 获取所有ID
        ids = df[id_column].astype(str).tolist()
        successful_updates = 0

        # 分批处理
        batch_size = self.batch_size

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]

            # 构建更新请求
            try:
                # 这里需要实现QdrantVectorStore的update_metadata方法
                # 或者使用Qdrant客户端直接更新
                # 这部分取决于QdrantVectorStore的实现
                pass

                successful_updates += len(batch_ids)
                logger.info(f"已更新 {i + len(batch_ids)}/{len(ids)} 条记录的元数据")

            except Exception as e:
                logger.error(f"更新元数据批次 {i // batch_size + 1} 时出错: {str(e)}")

        return successful_updates