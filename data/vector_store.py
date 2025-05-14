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

    # 针对时间戳处理的额外方法

    def _handle_timestamp_filter(self, filter_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        特别处理时间戳过滤条件

        Args:
            filter_conditions: 原始过滤条件

        Returns:
            修改后的过滤条件
        """
        import logging
        import copy
        logger = logging.getLogger(__name__)

        # 创建一个副本，避免修改原始过滤条件
        modified_filters = copy.deepcopy(filter_conditions)

        # 检查是否有时间戳过滤条件
        timestamp_fields = ["metadata.timestamp", "timestamp"]

        for field in timestamp_fields:
            if field in modified_filters:
                condition = modified_filters[field]

                # 如果是范围查询
                if isinstance(condition, dict) and any(k.startswith('$') for k in condition.keys()):
                    logger.info(f"检测到时间戳过滤条件: {field} -> {condition}")

                    # 确保时间格式正确（移除毫秒和时区信息如果有问题）
                    for op in ['$gte', '$gt', '$lte', '$lt', '$eq']:
                        if op in condition:
                            time_str = condition[op]
                            # 如果是字符串，尝试标准化时间格式
                            if isinstance(time_str, str):
                                # 尝试移除任何毫秒部分（如果导致问题）
                                # time_str = time_str.split('.')[0] + 'Z'
                                # 或者尝试替换时区格式
                                # time_str = time_str.replace('+00:00', 'Z')

                                # 更新条件
                                condition[op] = time_str
                                logger.info(f"标准化时间格式: {op} -> {time_str}")

        # 特殊处理: 如果有metadata.timestamp但Qdrant期望timestamp
        if "metadata.timestamp" in modified_filters and "timestamp" not in modified_filters:
            # 尝试复制条件到不同路径
            modified_filters["timestamp"] = modified_filters["metadata.timestamp"]
            logger.info(f"复制时间戳条件到备选路径: metadata.timestamp -> timestamp")

        return modified_filters

    # 修复 search_similar 方法使用正确的 Qdrant API

    def search_similar(
            self,
            query_vector: List[float],
            filter_conditions: Optional[Dict[str, Any]] = None,
            limit: int = 10,
            score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        搜索相似向量 - 修复 Qdrant 客户端 API 调用
        """
        import logging
        import json
        import requests
        from typing import Dict, List, Any, Optional

        logger = logging.getLogger(__name__)

        try:
            # 预处理过滤条件 - 特别处理时间戳
            if filter_conditions:
                filter_conditions = self._handle_timestamp_filter(filter_conditions)
                logger.info(f"处理后的过滤条件: {filter_conditions}")

            # 将过滤条件转换为Qdrant兼容格式
            qdrant_filter = None
            if filter_conditions:
                qdrant_filter = self._convert_filter_to_qdrant_format(filter_conditions)
                logger.info(f"转换后的Qdrant过滤条件: {qdrant_filter}")

            # 尝试多种搜索方法
            results = []

            # 方法1: 直接使用 REST API (绕过客户端可能的问题)
            try:
                # 构建 REST API 请求
                api_url = f"http://{self.host}:{self.port}/collections/{self.collection_name}/points/search"

                # 准备请求体
                request_body = {
                    "vector": query_vector,
                    "limit": limit,
                    "with_payload": True,
                    "with_vectors": False
                }

                # 添加过滤条件（如果有）
                if qdrant_filter:
                    request_body["filter"] = qdrant_filter

                logger.info(f"尝试方法1: 使用 REST API 直接调用，URL: {api_url}")
                response = requests.post(
                    api_url,
                    json=request_body,
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 200:
                    # 解析响应
                    response_data = response.json()
                    if "result" in response_data:
                        points = response_data["result"]

                        # 处理结果
                        for point in points:
                            result = {
                                "id": point.get("id"),
                                "score": float(point.get("score", 0.0)),
                                "payload": point.get("payload", {})
                            }
                            results.append(result)

                        logger.info(f"方法1成功，找到 {len(results)} 条记录")
                        return results
                else:
                    logger.warning(f"REST API 调用失败: {response.status_code} - {response.text}")
            except Exception as e:
                error_msg = f"方法1失败: {str(e)}"
                logger.warning(error_msg)

            # 方法2: 使用 Python 客户端但不使用过滤器
            if not results:
                try:
                    # 准备搜索参数（不带过滤器）
                    search_params = {
                        "collection_name": self.collection_name,
                        "query_vector": query_vector,
                        "limit": limit,
                        "with_payload": True
                    }

                    logger.info(f"尝试方法2: 使用 Python 客户端不带过滤器")
                    search_result = self.client.search(**search_params)

                    # 处理结果
                    for point in search_result:
                        result = {
                            "id": point.id,
                            "score": float(point.score),
                            "payload": point.payload
                        }
                        results.append(result)

                    logger.info(f"方法2成功，找到 {len(results)} 条记录")

                    # 手动过滤结果
                    if filter_conditions and len(results) > 0:
                        filtered_results = []

                        # 检查时间戳条件
                        start_time = None
                        end_time = None

                        if "metadata.timestamp" in filter_conditions:
                            timestamp_condition = filter_conditions["metadata.timestamp"]
                            if "$gte" in timestamp_condition:
                                start_time = timestamp_condition["$gte"]
                            if "$lte" in timestamp_condition:
                                end_time = timestamp_condition["$lte"]

                        if start_time or end_time:
                            logger.info(f"手动过滤时间戳: 开始={start_time}, 结束={end_time}")

                            for result in results:
                                # 检查结果是否在时间范围内
                                include = True
                                payload = result.get("payload", {})
                                metadata = payload.get("metadata", {})

                                # 尝试获取时间戳 (从不同可能的位置)
                                timestamp = None
                                if "timestamp" in metadata:
                                    timestamp = metadata["timestamp"]
                                elif "timestamp" in payload:
                                    timestamp = payload["timestamp"]

                                # 如果找到时间戳，检查范围
                                if timestamp:
                                    if start_time and timestamp < start_time:
                                        include = False
                                    if end_time and timestamp > end_time:
                                        include = False

                                if include:
                                    filtered_results.append(result)

                            logger.info(f"手动过滤后剩余 {len(filtered_results)} 条记录")
                            return filtered_results

                    # 如果没有时间戳过滤，返回所有结果
                    return results
                except Exception as e:
                    error_msg = f"方法2失败: {str(e)}"
                    logger.warning(error_msg)

            return results

        except Exception as e:
            logger.error(f"向量搜索总体失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def _convert_filter_to_qdrant_format(self, filter_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        将MongoDB风格过滤条件转换为Qdrant兼容的格式

        Args:
            filter_conditions: MongoDB风格的过滤条件

        Returns:
            Qdrant兼容格式的过滤条件
        """
        import logging
        logger = logging.getLogger(__name__)

        # 初始化Qdrant过滤条件
        qdrant_filter = {"must": []}

        # 遍历所有过滤条件
        for field, condition in filter_conditions.items():
            if isinstance(condition, dict) and any(k.startswith('$') for k in condition.keys()):
                # 处理范围查询 (MongoDB风格 -> Qdrant风格)
                range_condition = {}

                # 转换MongoDB操作符到Qdrant操作符
                operator_mapping = {
                    "$gte": "gte",  # 大于等于
                    "$gt": "gt",  # 大于
                    "$lte": "lte",  # 小于等于
                    "$lt": "lt",  # 小于
                    "$eq": "value"  # 等于 (Qdrant使用match.value)
                }

                for mongo_op, value in condition.items():
                    if mongo_op in operator_mapping:
                        qdrant_op = operator_mapping[mongo_op]

                        # 特殊处理$eq
                        if mongo_op == "$eq":
                            qdrant_filter["must"].append({
                                "key": field,
                                "match": {
                                    "value": value
                                }
                            })
                        else:
                            # 添加到范围条件
                            if "range" not in range_condition:
                                range_condition["range"] = {}
                            range_condition["range"][qdrant_op] = value

                # 如果有范围条件，添加到过滤器
                if range_condition.get("range"):
                    qdrant_filter["must"].append({
                        "key": field,
                        "range": range_condition["range"]
                    })

                logger.debug(f"添加范围过滤条件: {field} -> {range_condition}")
            elif isinstance(condition, list):
                # 处理列表查询 (IN查询) - MongoDB的 $in 对应 Qdrant的 match.any
                qdrant_filter["must"].append({
                    "key": field,
                    "match": {
                        "any": condition
                    }
                })

                logger.debug(f"添加列表过滤条件: {field} -> {condition}")
            else:
                # 处理精确匹配
                qdrant_filter["must"].append({
                    "key": field,
                    "match": {
                        "value": condition
                    }
                })

                logger.debug(f"添加精确匹配过滤条件: {field} -> {condition}")

        # 如果没有过滤条件，返回None
        if not qdrant_filter["must"]:
            return None

        return qdrant_filter

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