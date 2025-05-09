import logging
import time
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Qdrant向量数据库连接器"""

    def __init__(
            self,
            host: str = "localhost",
            port: int = 6333,
            grpc_port: int = 6334,
            api_key: Optional[str] = None,
            prefer_grpc: bool = True,
            timeout: float = 60.0,
            collection_name: str = "log_vectors",
            vector_size: int = 1024  # bge-m3向量维度
    ):
        """
        初始化Qdrant连接器

        Args:
            host: Qdrant主机地址
            port: Qdrant HTTP端口
            grpc_port: Qdrant gRPC端口
            api_key: API密钥（如果需要）
            prefer_grpc: 是否优先使用gRPC
            timeout: 连接超时时间
            collection_name: 集合名称
            vector_size: 向量维度
        """
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.collection_name = collection_name
        self.vector_size = vector_size

        # 初始化Qdrant客户端
        try:
            self.client = QdrantClient(
                url=host,
                port=port,
                grpc_port=grpc_port if prefer_grpc else None,
                api_key=api_key,
                timeout=timeout
            )
            logger.info(f"成功连接到Qdrant: {host}:{port}")

        except Exception as e:
            logger.error(f"连接Qdrant时出错: {str(e)}")
            raise

    def ensure_collection_exists(self, recreate: bool = False) -> bool:
        """
        确保集合存在，如果不存在则创建

        Args:
            recreate: 如果为True，则重新创建集合

        Returns:
            是否成功
        """
        try:
            # 检查集合是否存在
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)

            # 如果需要重新创建，则删除现有集合
            if collection_exists and recreate:
                logger.info(f"删除现有集合: {self.collection_name}")
                self.client.delete_collection(collection_name=self.collection_name)
                collection_exists = False

            # 如果集合不存在，创建新集合
            if not collection_exists:
                logger.info(f"创建新集合: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=self.vector_size,
                        distance=qdrant_models.Distance.COSINE
                    ),
                    optimizers_config=qdrant_models.OptimizersConfigDiff(
                        indexing_threshold=20000  # 设置较大的索引阈值以减少内存使用
                    )
                )

                # 创建索引以加快搜索
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="metadata.timestamp",
                    field_schema=qdrant_models.PayloadSchemaType.DATETIME
                )

                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="metadata.log_level",
                    field_schema=qdrant_models.PayloadSchemaType.KEYWORD
                )

                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="metadata.cluster_id",
                    field_schema=qdrant_models.PayloadSchemaType.INTEGER
                )

                logger.info(f"集合 {self.collection_name} 创建完成")
            else:
                logger.info(f"集合 {self.collection_name} 已存在")

            return True

        except Exception as e:
            logger.error(f"确保集合存在时出错: {str(e)}")
            return False

    def add_vectors(
            self,
            ids: List[str],
            vectors: List[List[float]],
            payloads: List[Dict[str, Any]],
            batch_size: int = 100
    ) -> bool:
        """
        批量添加向量到Qdrant

        Args:
            ids: 向量ID列表
            vectors: 向量列表
            payloads: 负载数据列表
            batch_size: 批量大小

        Returns:
            是否成功
        """
        if not ids or not vectors or not payloads:
            logger.warning("没有数据需要添加到Qdrant")
            return True

        if len(ids) != len(vectors) or len(vectors) != len(payloads):
            logger.error("IDs、向量和负载数据长度不匹配")
            return False

        try:
            # 确保集合存在
            if not self.ensure_collection_exists():
                return False

            total_records = len(ids)
            logger.info(f"开始批量添加 {total_records} 条记录到Qdrant")

            # 批量处理
            for i in range(0, total_records, batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_vectors = vectors[i:i + batch_size]
                batch_payloads = payloads[i:i + batch_size]

                # 创建点对象
                points = [
                    qdrant_models.PointStruct(
                        id=str(id),  # 确保ID是字符串
                        vector=vector,
                        payload=payload
                    )
                    for id, vector, payload in zip(batch_ids, batch_vectors, batch_payloads)
                ]

                # 批量插入
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )

                logger.info(f"已添加 {i + len(batch_ids)}/{total_records} 条记录")

            logger.info(f"成功添加 {total_records} 条记录到Qdrant")
            return True

        except Exception as e:
            logger.error(f"添加向量到Qdrant时出错: {str(e)}")
            return False

    def search_similar(
            self,
            query_vector: List[float],
            filter_conditions: Optional[Dict[str, Any]] = None,
            limit: int = 10,
            score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        搜索与查询向量相似的记录

        Args:
            query_vector: 查询向量
            filter_conditions: 过滤条件
            limit: 返回结果数量限制
            score_threshold: 相似度分数阈值

        Returns:
            相似记录列表
        """
        try:
            # 构建过滤条件
            filter_param = None
            if filter_conditions:
                filter_param = self._build_filter(filter_conditions)

            # 执行搜索
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=filter_param,
                limit=limit,
                score_threshold=score_threshold
            )

            # 转换结果
            results = []
            for scored_point in search_result:
                result = {
                    "id": scored_point.id,
                    "score": scored_point.score,
                    "payload": scored_point.payload
                }
                results.append(result)

            logger.info(f"搜索完成，找到 {len(results)} 条匹配记录")
            return results

        except Exception as e:
            logger.error(f"搜索Qdrant时出错: {str(e)}")
            return []

    def _build_filter(self, conditions: Dict[str, Any]) -> qdrant_models.Filter:
        """
        构建Qdrant过滤条件

        Args:
            conditions: 过滤条件字典

        Returns:
            Qdrant过滤器对象
        """
        import datetime
        import re

        filter_conditions = []

        # 时间戳字段模式，用于识别时间相关字段
        timestamp_pattern = re.compile(r'timestamp|time|date', re.IGNORECASE)

        for field, value in conditions.items():
            # 处理不同类型的条件
            if isinstance(value, dict):
                # 范围条件 {"metadata.timestamp": {"$gte": "2023-01-01", "$lte": "2023-12-31"}}
                for op, op_value in value.items():
                    # 检查是否需要转换时间字符串
                    is_timestamp_field = timestamp_pattern.search(field) is not None
                    converted_value = op_value

                    # 对时间字符串进行转换
                    if is_timestamp_field and isinstance(op_value, str):
                        try:
                            # 尝试将ISO时间字符串转换为Unix时间戳（秒）
                            dt = datetime.datetime.fromisoformat(op_value.replace('Z', '+00:00'))
                            converted_value = dt.timestamp()  # 转换为数值时间戳
                            logger.debug(f"将时间字符串 '{op_value}' 转换为时间戳 {converted_value}")
                        except Exception as e:
                            logger.warning(f"时间戳转换失败 '{op_value}': {str(e)}")

                    # 根据操作符构建过滤条件
                    if op == "$gte":
                        filter_conditions.append(
                            qdrant_models.FieldCondition(
                                key=field,
                                range=qdrant_models.Range(
                                    gte=converted_value
                                )
                            )
                        )
                    elif op == "$lte":
                        filter_conditions.append(
                            qdrant_models.FieldCondition(
                                key=field,
                                range=qdrant_models.Range(
                                    lte=converted_value
                                )
                            )
                        )
                    elif op == "$gt":
                        filter_conditions.append(
                            qdrant_models.FieldCondition(
                                key=field,
                                range=qdrant_models.Range(
                                    gt=converted_value
                                )
                            )
                        )
                    elif op == "$lt":
                        filter_conditions.append(
                            qdrant_models.FieldCondition(
                                key=field,
                                range=qdrant_models.Range(
                                    lt=converted_value
                                )
                            )
                        )
                    elif op == "$eq":
                        filter_conditions.append(
                            qdrant_models.FieldCondition(
                                key=field,
                                match=qdrant_models.MatchValue(value=converted_value)
                            )
                        )
                    elif op == "$ne":
                        # 不等于需要使用must_not
                        filter_conditions.append(
                            qdrant_models.FieldCondition(
                                key=field,
                                match=qdrant_models.MatchValue(value=converted_value),
                                must_not=True
                            )
                        )
                    # 可以添加其他操作符支持

            elif isinstance(value, list):
                # 列表条件 {"metadata.log_level": ["ERROR", "FATAL"]}
                filter_conditions.append(
                    qdrant_models.FieldCondition(
                        key=field,
                        match=qdrant_models.MatchAny(any=value)
                    )
                )
            else:
                # 单值条件 {"metadata.cluster_id": 5}
                # 检查是否是时间戳值需要转换
                is_timestamp_field = timestamp_pattern.search(field) is not None
                converted_value = value

                if is_timestamp_field and isinstance(value, str):
                    try:
                        dt = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
                        converted_value = dt.timestamp()  # 转换为数值时间戳
                    except Exception as e:
                        logger.warning(f"单值时间戳转换失败 '{value}': {str(e)}")

                filter_conditions.append(
                    qdrant_models.FieldCondition(
                        key=field,
                        match=qdrant_models.MatchValue(value=converted_value)
                    )
                )

        # 如果有多个条件，构建复合过滤器
        if len(filter_conditions) > 1:
            return qdrant_models.Filter(
                must=filter_conditions
            )
        elif len(filter_conditions) == 1:
            return qdrant_models.Filter(
                must=[filter_conditions[0]]
            )
        else:
            return None