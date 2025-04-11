# retrieval/hybrid_search.py (优化版)
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import requests

from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from models.embedding import BaseEmbeddingModel
from data.es_connector import ESConnector
from data.vector_store import QdrantVectorStore
from utils import get_logger, clean_text

logger = get_logger(__name__)


class HybridSearch:
    """
    混合搜索类，结合Elasticsearch的BM25、Qdrant的向量搜索和重排序模型

    主要功能：
    1. 文本搜索 - 使用ES的BM25进行关键词搜索
    2. 语义搜索 - 使用Qdrant进行向量相似度搜索
    3. 重排序 - 使用重排序模型进一步优化结果
    4. 混合排序 - 结合多种搜索结果进行排序
    5. 高级过滤 - 支持按时间、日志级别、服务等过滤
    """

    def __init__(
            self,
            es_connector: ESConnector,
            vector_store: QdrantVectorStore,
            embedding_model: BaseEmbeddingModel,
            rerank_url: str = "http://localhost:8091/rerank",
            es_weight: float = 0.3,
            vector_weight: float = 0.5,
            rerank_weight: float = 0.2,
            use_rerank: bool = True,
            max_es_results: int = 50,
            max_vector_results: int = 50,
            rerank_results: int = 30,
            final_results_limit: int = 20
    ):
        """
        初始化混合搜索

        Args:
            es_connector: ES连接器
            vector_store: 向量存储
            embedding_model: 嵌入模型
            rerank_url: 重排序服务URL
            es_weight: ES搜索结果权重
            vector_weight: 向量搜索结果权重
            rerank_weight: 重排序结果权重
            use_rerank: 是否使用重排序
            max_es_results: ES最大结果数
            max_vector_results: 向量搜索最大结果数
            rerank_results: 重排序的最大结果数
            final_results_limit: 最终返回结果数量限制
        """
        self.es_connector = es_connector
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.rerank_url = rerank_url

        self.es_weight = es_weight
        self.vector_weight = vector_weight
        self.rerank_weight = rerank_weight
        self.use_rerank = use_rerank

        self.max_es_results = max_es_results
        self.max_vector_results = max_vector_results
        self.rerank_results = rerank_results
        self.final_results_limit = final_results_limit

        logger.info(f"混合搜索初始化，ES权重: {es_weight}, 向量权重: {vector_weight}, 重排序权重: {rerank_weight}")

    def search(
            self,
            query: str,
            filters: Optional[Dict[str, Any]] = None,
            search_processed_index: bool = True,
            include_vectors: bool = True,
            time_range: Optional[Tuple[datetime, datetime]] = None,
            services: Optional[List[str]] = None,
            log_levels: Optional[List[str]] = None,
            use_error_cache: bool = True,
            use_rerank: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        执行混合搜索

        Args:
            query: 搜索查询
            filters: 附加过滤条件
            search_processed_index: 是否搜索处理后的索引（含IK分词）
            include_vectors: 是否包含向量搜索
            time_range: 时间范围过滤
            services: 服务名称过滤
            log_levels: 日志级别过滤
            use_error_cache: 是否使用错误缓存
            use_rerank: 是否使用重排序（覆盖实例设置）

        Returns:
            搜索结果字典
        """
        if not query:
            return {"status": "error", "message": "查询不能为空", "results": []}

        logger.info(f"执行混合搜索: '{query}'")
        start_time = datetime.now()

        # 确定是否使用重排序
        apply_rerank = self.use_rerank if use_rerank is None else use_rerank

        # 准备过滤条件
        es_filters = self._prepare_es_filters(filters, time_range, services, log_levels)
        vector_filters = self._prepare_vector_filters(filters, time_range, services, log_levels)

        # 执行向量搜索
        vector_results = []
        if include_vectors:
            vector_results = self._search_vectors(query, vector_filters)

        # 执行ES搜索
        es_results = self._search_es(query, es_filters, search_processed_index)

        # 合并初步结果以准备重排序
        initial_results = self._merge_initial_results(es_results, vector_results)

        # 执行重排序（如果启用）
        reranked_results = []
        if apply_rerank and initial_results:
            reranked_results = self._rerank_results(query, initial_results[:self.rerank_results])

        # 最终混合结果
        combined_results = self._merge_final_results(query, es_results, vector_results, reranked_results)

        search_time = (datetime.now() - start_time).total_seconds()

        return {
            "status": "success",
            "query": query,
            "es_results_count": len(es_results),
            "vector_results_count": len(vector_results),
            "reranked_results_count": len(reranked_results),
            "combined_results_count": len(combined_results),
            "search_time": search_time,
            "results": combined_results
        }

    def _merge_initial_results(
            self,
            es_results: List[Dict[str, Any]],
            vector_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        合并ES和向量结果，准备重排序

        Args:
            es_results: ES搜索结果
            vector_results: 向量搜索结果

        Returns:
            合并的结果列表
        """
        # 创建ID到结果的映射
        result_map = {}

        # 处理ES结果
        for result in es_results:
            result_id = result["id"]
            result_map[result_id] = result

        # 处理向量结果，避免重复
        for result in vector_results:
            result_id = result["id"]
            if result_id not in result_map:
                result_map[result_id] = result

        # 转换为列表
        return list(result_map.values())

    def _rerank_results(
            self,
            query: str,
            results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        使用重排序模型重新排序结果

        Args:
            query: 搜索查询
            results: 初步搜索结果

        Returns:
            重排序后的结果
        """
        if not results:
            return []

        try:
            # 准备重排序输入格式
            documents = []
            for result in results:
                # 提取消息作为内容
                content = result.get("message", "")
                # 构建文档对象
                doc = {
                    "id": result.get("id", ""),
                    "content": content,
                    # 保留原始分数用于后续混合
                    "es_score": result.get("es_score", 0.0),
                    "vector_score": result.get("vector_score", 0.0),
                    # 保存原始结果
                    "original_result": result
                }
                documents.append(doc)

            # 调用重排序API
            payload = {
                "query": query,
                "documents": documents,
                "top_k": len(documents)  # 保留所有结果
            }

            response = requests.post(
                self.rerank_url,
                json=payload,
                timeout=30
            )

            if response.status_code != 200:
                logger.warning(f"重排序API返回错误状态码: {response.status_code}")
                return []

            response_data = response.json()
            reranked_docs = response_data.get("results", [])

            # 处理重排序结果
            processed_results = []
            for doc in reranked_docs:
                # 获取原始结果
                original_result = doc.get("original_result", {})
                # 添加重排序分数
                rerank_score = doc.get("rerank_score", 0.0)

                # 合并结果
                processed_result = {
                    **original_result,
                    "rerank_score": rerank_score
                }
                processed_results.append(processed_result)

            logger.info(f"重排序返回 {len(processed_results)} 条结果")
            return processed_results

        except Exception as e:
            logger.error(f"重排序失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def _merge_final_results(
            self,
            query: str,
            es_results: List[Dict[str, Any]],
            vector_results: List[Dict[str, Any]],
            reranked_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        最终合并所有搜索结果

        Args:
            query: 原始查询
            es_results: ES搜索结果
            vector_results: 向量搜索结果
            reranked_results: 重排序结果

        Returns:
            合并后的结果列表
        """
        # 创建ID到结果的映射
        result_map = {}

        # 处理ES结果
        for result in es_results:
            result_id = result["id"]
            result_map[result_id] = {
                **result,
                "es_score": result["score"],
                "vector_score": 0.0,
                "rerank_score": 0.0,
                "final_score": self.es_weight * result["score"]  # 初始得分只包含ES部分
            }

        # 处理向量结果
        for result in vector_results:
            result_id = result["id"]

            if result_id in result_map:
                # 如果结果已存在，更新分数
                result_map[result_id]["vector_score"] = result["score"]
                result_map[result_id]["final_score"] += self.vector_weight * result["score"]
            else:
                # 新结果
                result_map[result_id] = {
                    **result,
                    "es_score": 0.0,
                    "vector_score": result["score"],
                    "rerank_score": 0.0,
                    "final_score": self.vector_weight * result["score"]  # 初始得分只包含向量部分
                }

        # 处理重排序结果
        for result in reranked_results:
            result_id = result["id"]

            if result_id in result_map:
                # 更新重排序分数
                result_map[result_id]["rerank_score"] = result["rerank_score"]
                result_map[result_id]["final_score"] += self.rerank_weight * result["rerank_score"]
            else:
                # 这种情况理论上不应该发生，因为重排序基于初始结果
                result_map[result_id] = {
                    **result,
                    "es_score": result.get("es_score", 0.0),
                    "vector_score": result.get("vector_score", 0.0),
                    "rerank_score": result["rerank_score"],
                    "final_score": self.rerank_weight * result["rerank_score"]
                }

        # 转换为列表并排序
        combined_results = list(result_map.values())
        combined_results.sort(key=lambda x: x["final_score"], reverse=True)

        # 限制结果数量
        combined_results = combined_results[:self.final_results_limit]

        # 添加排名字段
        for i, result in enumerate(combined_results):
            result["rank"] = i + 1

        # 增加相关性分析
        self._add_relevance_info(query, combined_results)

        return combined_results