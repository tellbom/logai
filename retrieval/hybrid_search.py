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

    def set_weights(self, es_weight: float, vector_weight: float):
        """设置混合搜索的权重参数

        Args:
            es_weight: Elasticsearch搜索的权重
            vector_weight: 向量搜索的权重
        """
        # 校验权重
        if 0.0 <= es_weight <= 1.0 and 0.0 <= vector_weight <= 1.0:
            self.es_weight = es_weight
            self.vector_weight = vector_weight
            logger.info(f"已更新搜索权重: ES={es_weight}, Vector={vector_weight}")
        else:
            logger.warning(f"权重设置无效，需在0-1范围内: ES={es_weight}, Vector={vector_weight}")
            logger.warning(f"保持原有权重: ES={self.es_weight}, Vector={self.vector_weight}")

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

    def _merge_final_results(self, query: str, es_results: List[Dict], vector_results: List[Dict],
                             reranked_results: List[Dict]) -> List[Dict]:
        """改进的结果合并算法"""
        # 创建基础合并结果
        result_map = self._create_base_result_map(es_results, vector_results, reranked_results)
        combined_results = list(result_map.values())

        if not combined_results:
            return []

        # 计算动态权重
        es_weight, vector_weight, rerank_weight = self._get_dynamic_weights(query, es_results, vector_results)

        # 计算TF-IDF相关性
        tfidf_scores = self._calculate_tfidf_relevance(query, combined_results)

        # 计算上下文分数
        context_scores = self._calculate_context_score(combined_results)

        # 整合所有分数
        for result in combined_results:
            # 基础分数
            base_score = (
                    es_weight * result.get("es_score", 0.0) +
                    vector_weight * result.get("vector_score", 0.0) +
                    rerank_weight * result.get("rerank_score", 0.0)
            )

            # 日志级别权重
            level_importance = self._get_level_importance(result.get("log_level", "INFO"))

            # TF-IDF相关性
            tfidf_score = tfidf_scores.get(result["id"], 0.0)

            # 上下文相关性
            context_score = context_scores.get(result["id"], 1.0)

            # 最终分数计算
            result["final_score"] = base_score * (
                        0.7 + 0.1 * level_importance + 0.1 * tfidf_score + 0.1 * context_score)

        # 排序和限制结果
        combined_results.sort(key=lambda x: x["final_score"], reverse=True)
        combined_results = combined_results[:self.final_results_limit]

        # 添加排名和相关性信息
        for i, result in enumerate(combined_results):
            result["rank"] = i + 1

        self._add_relevance_info(query, combined_results)

        return combined_results

    def _prepare_es_filters(
            self,
            filters: Optional[Dict[str, Any]] = None,
            time_range: Optional[Tuple[datetime, datetime]] = None,
            services: Optional[List[str]] = None,
            log_levels: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        准备ES过滤条件

        Args:
            filters: 自定义过滤条件
            time_range: 时间范围
            services: 服务列表
            log_levels: 日志级别列表

        Returns:
            ES过滤条件列表
        """
        es_filters = []

        # 时间范围过滤
        if time_range:
            start_time, end_time = time_range
            time_filter = {
                "range": {
                    "@timestamp": {
                        "gte": start_time.isoformat(),
                        "lte": end_time.isoformat()
                    }
                }
            }
            es_filters.append(time_filter)

        # 服务过滤
        if services:
            service_filter = {
                "terms": {
                    "service_name": services
                }
            }
            es_filters.append(service_filter)

        # 日志级别过滤
        if log_levels:
            level_filter = {
                "terms": {
                    "log_level": log_levels
                }
            }
            es_filters.append(level_filter)

        # 添加自定义过滤条件
        if filters:
            for field, value in filters.items():
                if isinstance(value, list):
                    filter_item = {"terms": {field: value}}
                elif isinstance(value, dict) and ("gte" in value or "lte" in value):
                    filter_item = {"range": {field: value}}
                else:
                    filter_item = {"term": {field: value}}

                es_filters.append(filter_item)

        return es_filters

    def _prepare_vector_filters(
            self,
            filters: Optional[Dict[str, Any]] = None,
            time_range: Optional[Tuple[datetime, datetime]] = None,
            services: Optional[List[str]] = None,
            log_levels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        准备向量搜索过滤条件

        Args:
            filters: 自定义过滤条件
            time_range: 时间范围
            services: 服务列表
            log_levels: 日志级别列表

        Returns:
            向量搜索过滤条件
        """
        vector_filters = {}

        # 时间范围过滤
        if time_range:
            start_time, end_time = time_range
            vector_filters["metadata.timestamp"] = {
                "$gte": start_time.isoformat(),
                "$lte": end_time.isoformat()
            }

        # 服务过滤
        if services:
            vector_filters["metadata.service_name"] = services

        # 日志级别过滤
        if log_levels:
            vector_filters["metadata.log_level"] = log_levels

        # 添加自定义过滤条件
        if filters:
            for field, value in filters.items():
                vector_key = f"metadata.{field}"
                if isinstance(value, dict) and ("gte" in value or "lte" in value):
                    range_filter = {}
                    if "gte" in value:
                        range_filter["$gte"] = value["gte"]
                    if "lte" in value:
                        range_filter["$lte"] = value["lte"]
                    vector_filters[vector_key] = range_filter
                else:
                    vector_filters[vector_key] = value

        return vector_filters

    def _search_vectors(
            self,
            query: str,
            filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        执行向量搜索

        Args:
            query: 搜索查询
            filters: 过滤条件

        Returns:
            向量搜索结果列表
        """
        try:
            # 清理查询文本
            cleaned_query = clean_text(query)

            # 生成查询向量
            query_vector = self.embedding_model.embed([cleaned_query])[0]

            # 执行向量搜索
            vector_results = self.vector_store.search_similar(
                query_vector=query_vector,
                filter_conditions=filters,
                limit=self.max_vector_results,
                score_threshold=0.5  # 相似度阈值
            )

            # 处理结果
            processed_results = []

            for result in vector_results:
                # 获取元数据
                metadata = result.get("payload", {}).get("metadata", {})

                processed_result = {
                    "id": result.get("id", ""),
                    "score": float(result.get("score", 0.0)),
                    "source": "vector",
                    "log_level": metadata.get("log_level", ""),
                    "timestamp": metadata.get("timestamp", ""),
                    "message": metadata.get("original_text", ""),
                    "service_name": metadata.get("service_name", ""),
                    "template": metadata.get("template", ""),
                    "cluster_id": metadata.get("cluster_id", ""),
                    "metadata": metadata
                }

                processed_results.append(processed_result)

            logger.info(f"向量搜索返回 {len(processed_results)} 条结果")
            return processed_results

        except Exception as e:
            logger.error(f"向量搜索失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def _search_es(
            self,
            query: str,
            filters: Optional[List[Dict[str, Any]]] = None,
            search_processed_index: bool = True
    ) -> List[Dict[str, Any]]:
        """
        执行ES搜索

        Args:
            query: 搜索查询
            filters: 过滤条件
            search_processed_index: 是否搜索处理后的索引

        Returns:
            ES搜索结果列表
        """
        try:
            # 构建ES查询
            es_query = {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "message": {
                                    "query": query,
                                     "operator": "or",
                                     "minimum_should_match": "30%"
                                }
                            }
                        }
                    ]
                }
            }

            # 添加过滤条件
            if filters:
                es_query["bool"]["filter"] = filters

            # 确定搜索索引
            index_pattern = "logai-processed" if search_processed_index else self.es_connector.es_index_pattern

            # 执行搜索
            search_results = self.es_connector.es_client.search(
                index=index_pattern,
                body={
                    "query": es_query,
                    "size": self.max_es_results,
                    "sort": [
                        {"_score": {"order": "desc"}},
                        {"@timestamp": {"order": "desc"}}
                    ]
                }
            )

            # 处理结果
            processed_results = []

            for hit in search_results["hits"]["hits"]:
                source = hit["_source"]

                processed_result = {
                    "id": hit["_id"],
                    "score": float(hit["_score"]),
                    "source": "elasticsearch",
                    "log_level": source.get("log_level", ""),
                    "timestamp": source.get("@timestamp", ""),
                    "message": source.get("message", ""),
                    "service_name": source.get("service_name", ""),
                    "template": source.get("template", ""),
                    "cluster_id": source.get("cluster_id", ""),
                    "metadata": {k: v for k, v in source.items() if k not in ["message", "@timestamp"]}
                }

                processed_results.append(processed_result)

            logger.info(f"ES搜索返回 {len(processed_results)} 条结果")
            return processed_results

        except Exception as e:
            logger.error(f"ES搜索失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def _add_relevance_info(self, query: str, results: List[Dict[str, Any]]) -> None:
        """
        为搜索结果添加相关性信息

        Args:
            query: 搜索查询
            results: 搜索结果列表
        """
        if not results:
            return

        # 将查询拆分为关键词
        query_terms = set(query.lower().split())

        for result in results:
            # 获取消息文本
            message = result.get("message", "")

            # 计算关键词匹配度
            matched_terms = 0
            for term in query_terms:
                if term and len(term) > 2 and term in message.lower():
                    matched_terms += 1

            # 计算关键词覆盖率
            keyword_coverage = matched_terms / len(query_terms) if query_terms else 0

            # 添加相关性信息
            result["relevance"] = {
                "keyword_coverage": float(keyword_coverage),
                "matched_terms": matched_terms,
                "total_query_terms": len(query_terms)
            }

            # 简单分类相关性
            if result.get("final_score", 0) > 0.8:
                relevance_level = "high"
            elif result.get("final_score", 0) > 0.5:
                relevance_level = "medium"
            else:
                relevance_level = "low"

            result["relevance"]["level"] = relevance_level

    def _get_dynamic_weights(self, query: str, es_results: List[Dict], vector_results: List[Dict]) -> Tuple[
        float, float, float]:
        """根据查询特征动态调整权重"""
        query_lower = query.lower()

        # 如果查询更像是关键词搜索(短查询，包含特定术语)
        if len(query.split()) <= 3 or any(
                term in query_lower for term in ['error', 'exception', 'fail', '错误', '异常']):
            return 0.5, 0.3, 0.2  # 提高ES权重

        # 如果查询更像是语义问题(长句子，疑问句)
        elif len(query.split()) >= 5 or query_lower.endswith('?') or '什么' in query_lower or '如何' in query_lower:
            return 0.2, 0.6, 0.2  # 提高向量权重

        # 默认权重
        return self.es_weight, self.vector_weight, self.rerank_weight

    def _calculate_context_score(self, results: List[Dict]) -> Dict[str, float]:
        """计算上下文相关性分数"""
        # 提取时间戳并按时间排序
        timestamps = [(i, result.get("timestamp")) for i, result in enumerate(results) if result.get("timestamp")]
        timestamps.sort(key=lambda x: x[1])

        context_scores = {result["id"]: 1.0 for result in results}

        # 为时间接近的文档增加权重
        for i in range(len(timestamps) - 1):
            curr_idx, curr_time = timestamps[i]
            next_idx, next_time = timestamps[i + 1]

            # 如果两条日志时间接近(例如5分钟内)
            time_diff = (next_time - curr_time).total_seconds()
            if time_diff < 300:  # 5分钟
                boost = 1.0 - (time_diff / 300) * 0.2  # 时间越接近，提升越大
                context_scores[results[curr_idx]["id"]] += boost
                context_scores[results[next_idx]["id"]] += boost

        return context_scores

    def _calculate_tfidf_relevance(self, query: str, results: List[Dict]) -> Dict[str, float]:
        """使用TF-IDF计算查询与结果的相关性"""
        from sklearn.feature_extraction.text import TfidfVectorizer

        # 提取所有消息文本
        docs = [result.get("message", "") for result in results]
        if not docs:
            return {}

        # 添加查询到文档集合
        all_docs = [query] + docs

        # TF-IDF向量化
        vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_docs)

        # 计算查询与每个文档的余弦相似度
        query_vector = tfidf_matrix[0:1]
        doc_vectors = tfidf_matrix[1:]

        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_vector, doc_vectors).flatten()

        # 创建ID到相似度的映射
        tfidf_scores = {}
        for i, result in enumerate(results):
            tfidf_scores[result["id"]] = float(similarities[i])

        return tfidf_scores

    def _get_level_importance(self, log_level: str) -> float:
        """根据日志级别返回重要性权重"""
        importance_map = {
            "FATAL": 1.0,
            "ERROR": 0.9,
            "WARN": 0.7,
            "WARNING": 0.7,
            "INFO": 0.5,
            "DEBUG": 0.3,
            "TRACE": 0.2
        }
        return importance_map.get(log_level.upper(), 0.5)

    def _add_temporal_analysis(self, results: List[Dict]) -> None:
        """添加时序关系分析"""
        if len(results) < 2:
            return

        # 按时间排序
        time_sorted = sorted(results, key=lambda x: x.get("timestamp", ""))

        # 标记时间邻近的记录
        for i in range(len(time_sorted) - 1):
            curr = time_sorted[i]
            next_log = time_sorted[i + 1]

            if "timestamp" in curr and "timestamp" in next_log:
                try:
                    curr_time = datetime.fromisoformat(curr["timestamp"].replace("Z", "+00:00"))
                    next_time = datetime.fromisoformat(next_log["timestamp"].replace("Z", "+00:00"))

                    time_diff = (next_time - curr_time).total_seconds()

                    if time_diff < 60:  # 1分钟内
                        if "temporal_relations" not in curr:
                            curr["temporal_relations"] = []
                        if "temporal_relations" not in next_log:
                            next_log["temporal_relations"] = []

                        curr["temporal_relations"].append({"id": next_log["id"], "time_diff": time_diff})
                        next_log["temporal_relations"].append({"id": curr["id"], "time_diff": -time_diff})
                except:
                    pass

    def _create_base_result_map(
            self,
            es_results: List[Dict[str, Any]],
            vector_results: List[Dict[str, Any]],
            reranked_results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        创建基础结果映射，合并来自不同来源的结果

        Args:
            es_results: ES搜索结果
            vector_results: 向量搜索结果
            reranked_results: 重排序结果

        Returns:
            ID到结果的映射字典
        """
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

        return result_map