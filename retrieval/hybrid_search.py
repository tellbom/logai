# retrieval/hybrid_search.py
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

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
    混合搜索类，结合Elasticsearch的BM25和Qdrant的向量搜索

    主要功能：
    1. 文本搜索 - 使用ES的BM25进行关键词搜索
    2. 语义搜索 - 使用Qdrant进行向量相似度搜索
    3. 混合排序 - 结合两种搜索结果进行排序
    4. 高级过滤 - 支持按时间、日志级别、服务等过滤
    """

    def __init__(
            self,
            es_connector: ESConnector,
            vector_store: QdrantVectorStore,
            embedding_model: BaseEmbeddingModel,
            es_weight: float = 0.3,
            vector_weight: float = 0.7,
            max_es_results: int = 50,
            max_vector_results: int = 50,
            final_results_limit: int = 20
    ):
        """
        初始化混合搜索

        Args:
            es_connector: ES连接器
            vector_store: 向量存储
            embedding_model: 嵌入模型
            es_weight: ES搜索结果权重
            vector_weight: 向量搜索结果权重
            max_es_results: ES最大结果数
            max_vector_results: 向量搜索最大结果数
            final_results_limit: 最终返回结果数量限制
        """
        self.es_connector = es_connector
        self.vector_store = vector_store
        self.embedding_model = embedding_model

        self.es_weight = es_weight
        self.vector_weight = vector_weight
        self.max_es_results = max_es_results
        self.max_vector_results = max_vector_results
        self.final_results_limit = final_results_limit

        logger.info(f"混合搜索初始化，ES权重: {es_weight}, 向量权重: {vector_weight}")

    def search(
            self,
            query: str,
            filters: Optional[Dict[str, Any]] = None,
            search_processed_index: bool = True,
            include_vectors: bool = True,
            time_range: Optional[Tuple[datetime, datetime]] = None,
            services: Optional[List[str]] = None,
            log_levels: Optional[List[str]] = None,
            use_error_cache: bool = True
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

        Returns:
            搜索结果字典
        """
        if not query:
            return {"status": "error", "message": "查询不能为空", "results": []}

        logger.info(f"执行混合搜索: '{query}'")
        start_time = datetime.now()

        # 准备过滤条件
        es_filters = self._prepare_es_filters(filters, time_range, services, log_levels)
        vector_filters = self._prepare_vector_filters(filters, time_range, services, log_levels)

        # 执行向量搜索
        vector_results = []
        if include_vectors:
            vector_results = self._search_vectors(query, vector_filters)

        # 执行ES搜索
        es_results = self._search_es(query, es_filters, search_processed_index)

        # 混合结果
        combined_results = self._merge_results(query, es_results, vector_results)

        search_time = (datetime.now() - start_time).total_seconds()

        return {
            "status": "success",
            "query": query,
            "es_results_count": len(es_results),
            "vector_results_count": len(vector_results),
            "combined_results_count": len(combined_results),
            "search_time": search_time,
            "results": combined_results
        }

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
                                    "operator": "and"
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

    def _merge_results(
            self,
            query: str,
            es_results: List[Dict[str, Any]],
            vector_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        合并ES和向量搜索结果

        Args:
            query: 原始查询
            es_results: ES搜索结果
            vector_results: 向量搜索结果

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
                    "final_score": self.vector_weight * result["score"]  # 初始得分只包含向量部分
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

    def _add_relevance_info(
            self,
            query: str,
            results: List[Dict[str, Any]]
    ) -> None:
        """
        为结果添加相关性信息

        Args:
            query: 原始查询
            results: 结果列表
        """
        # 简单的相关性分析：高亮匹配的关键词
        try:
            query_terms = set(query.lower().split())

            for result in results:
                message = result.get("message", "").lower()

                # 计算包含的查询词数量
                matched_terms = [term for term in query_terms if term in message]
                result["matched_terms"] = matched_terms
                result["matched_term_count"] = len(matched_terms)
                result["relevant_context"] = self._extract_relevant_context(result["message"], matched_terms)
        except Exception as e:
            logger.error(f"添加相关性信息失败: {str(e)}")

    def _extract_relevant_context(
            self,
            message: str,
            matched_terms: List[str],
            context_size: int = 200
    ) -> str:
        """
        提取匹配项周围的相关上下文

        Args:
            message: 原始消息
            matched_terms: 匹配的术语
            context_size: 上下文大小（字符数）

        Returns:
            相关上下文
        """
        if not matched_terms or not message:
            return message

        # 找到第一个匹配项的位置
        message_lower = message.lower()
        first_match_pos = min(
            (message_lower.find(term) for term in matched_terms if term in message_lower),
            default=-1
        )

        if first_match_pos == -1:
            return message

        # 提取上下文
        start_pos = max(0, first_match_pos - context_size // 2)
        end_pos = min(len(message), first_match_pos + context_size // 2)

        context = message[start_pos:end_pos]

        # 如果截断了，添加省略号
        if start_pos > 0:
            context = "..." + context
        if end_pos < len(message):
            context = context + "..."

        return context

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

    def search_by_example(
            self,
            example_log: str,
            filters: Optional[Dict[str, Any]] = None,
            include_es: bool = True,
            limit: int = 20
    ) -> Dict[str, Any]:
        """
        通过示例日志搜索相似日志

        Args:
            example_log: 示例日志
            filters: 过滤条件
            include_es: 是否包含ES搜索
            limit: 结果数量限制

        Returns:
            搜索结果
        """
        if not example_log:
            return {"status": "error", "message": "示例日志不能为空", "results": []}

        try:
            # 向量搜索
            cleaned_example = clean_text(example_log)
            example_vector = self.embedding_model.embed([cleaned_example])[0]

            vector_filters = self._prepare_vector_filters(filters)

            vector_results = self.vector_store.search_similar(
                query_vector=example_vector,
                filter_conditions=vector_filters,
                limit=self.max_vector_results,
                score_threshold=0.7  # 更高的阈值用于示例搜索
            )

            # 如果需要，也执行ES搜索
            es_results = []
            if include_es:
                # 从示例中提取关键词
                keywords = example_log.split()[:10]  # 简单截取前10个词作为关键词
                query = " ".join(keywords)

                es_filters = self._prepare_es_filters(filters)
                es_results = self._search_es(query, es_filters)

            # 合并结果
            combined_results = self._merge_results(example_log, es_results, vector_results)

            # 限制结果数量
            combined_results = combined_results[:limit]

            return {
                "status": "success",
                "example": example_log,
                "es_results_count": len(es_results),
                "vector_results_count": len(vector_results),
                "combined_results_count": len(combined_results),
                "results": combined_results
            }

        except Exception as e:
            logger.error(f"示例搜索失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e), "results": []}

    def get_relevant_context(
            self,
            query: str,
            max_results: int = 5,
            token_limit: int = 4000
    ) -> Dict[str, Any]:
        """
        获取与查询相关的上下文，用于LLM问答

        Args:
            query: 用户查询
            max_results: 最大结果数量
            token_limit: 上下文令牌限制

        Returns:
            包含相关上下文的字典
        """
        # 执行搜索
        search_results = self.search(query, include_vectors=True)

        # 如果没有结果，返回空上下文
        if not search_results["results"]:
            return {
                "context": "",
                "sources": [],
                "query": query
            }

        # 提取最相关的结果
        top_results = search_results["results"][:max_results]

        # 构建上下文
        context_parts = []
        sources = []

        for i, result in enumerate(top_results):
            # 格式化日志记录
            timestamp = result.get("timestamp", "")
            log_level = result.get("log_level", "")
            message = result.get("message", "")
            service = result.get("service_name", "")

            context_part = f"[日志 {i + 1}] 时间: {timestamp}, 级别: {log_level}, 服务: {service}\n内容: {message}"
            context_parts.append(context_part)

            # 添加来源信息
            sources.append({
                "id": result.get("id", ""),
                "timestamp": timestamp,
                "service": service,
                "level": log_level,
                "score": result.get("final_score", 0)
            })

        # 组合上下文
        full_context = "\n\n".join(context_parts)

        # 确保不超过令牌限制（简单估计：1个中文字符约1.5个token，1个英文单词约1.5个token）
        # 这只是粗略估计，实际情况可能不同
        rough_token_count = len(full_context.split()) + sum(
            1.5 for char in full_context if '\u4e00' <= char <= '\u9fff')

        if rough_token_count > token_limit:
            # 简单截断，实际应用中可能需要更复杂的处理
            char_limit = int(token_limit * 0.6)  # 粗略转换为字符限制
            full_context = full_context[:char_limit] + "...(内容已截断)"

        return {
            "context": full_context,
            "sources": sources,
            "query": query
        }