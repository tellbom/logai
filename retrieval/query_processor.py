# retrieval/query_processor.py
import logging
import re
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import json

from models.embedding import BaseEmbeddingModel
from data.es_connector import ESConnector
from data.vector_store import QdrantVectorStore
from retrieval.hybrid_search import HybridSearch
from utils import get_logger, clean_text, extract_keywords

logger = get_logger(__name__)


class QueryProcessor:
    """
    查询处理器，处理用户查询并生成LLM提示词

    主要功能：
    1. 查询理解 - 分析用户查询意图和要素
    2. 查询扩展 - 扩展用户查询以提高召回率
    3. 提示词生成 - 为LLM生成结构化提示词
    4. 结果解析 - 处理搜索结果和LLM回答
    """

    def __init__(
            self,
            hybrid_search: HybridSearch,
            prompt_template: Optional[str] = None,
            max_context_logs: int = 5,
            token_limit: int = 4000,
            min_confidence: float = 0.6,
            field_mappings: Optional[Dict[str, str]] = None
    ):
        """
        初始化查询处理器

        Args:
            hybrid_search: 混合搜索对象
            prompt_template: 提示词模板
            max_context_logs: 最大上下文日志数量
            token_limit: 令牌限制
            min_confidence: 最小置信度
            field_mappings: 字段映射配置
        """
        self.hybrid_search = hybrid_search
        self.max_context_logs = max_context_logs
        self.token_limit = token_limit
        self.min_confidence = min_confidence

        # 设置字段映射
        self.field_mappings = field_mappings or {}
        self.message_field = self.field_mappings.get("message_field", "message")
        self.level_field = self.field_mappings.get("level_field", "log_level")
        self.service_field = self.field_mappings.get("service_field", "service_name")
        # timestamp_field 保持默认 "@timestamp"

        # 默认提示词模板
        self.default_prompt_template = """
    你是一个专业的日志分析助手，负责帮助用户分析和理解系统日志。
    请基于下面提供的日志信息回答用户的问题。如果日志中没有相关信息，请直接说明无法从提供的日志中找到答案。

    ### 相关日志信息:
    {context}

    ### 用户问题:
    {query}

    请分析这些日志并回答用户的问题。你的回答应该:
    1. 简洁明了，直接针对问题给出答案
    2. 必要时引用具体的日志内容作为依据
    3. 如果是错误分析，尽可能提供错误原因和可能的解决方案
    4. 如果日志信息不足以回答问题，请明确指出，不要猜测或编造信息
    """

        # 使用自定义模板或默认模板
        self.prompt_template = prompt_template or self.default_prompt_template

        logger.info("查询处理器初始化完成")


    def process_query(
            self,
            query: str,
            filters: Optional[Dict[str, Any]] = None,
            return_search_results: bool = False,
            return_prompt: bool = True,
            time_range: Optional[Tuple[datetime, datetime]] = None,
            services: Optional[List[str]] = None,
            log_levels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        处理用户查询

        Args:
            query: 用户查询
            filters: 过滤条件
            return_search_results: 是否返回搜索结果
            return_prompt: 是否返回提示词
            time_range: 时间范围
            services: 服务列表
            log_levels: 日志级别列表

        Returns:
            处理结果字典
        """
        if not query:
            return {"status": "error", "message": "查询不能为空"}

        start_time = datetime.now()

        try:
            # 分析查询意图
            query_info = self._analyze_query(query)

            # 扩展查询（如果需要）
            expanded_query = self._expand_query(query, query_info)

            # 执行搜索
            search_results = self.hybrid_search.search(
                query=expanded_query,
                filters=filters,
                time_range=time_range,
                services=services,
                log_levels=log_levels
            )

            # 构建上下文
            context_data = self._build_context(search_results["results"])

            # 生成提示词
            prompt = self._generate_prompt(query, context_data["context"])

            # 准备返回结果
            response = {
                "status": "success",
                "query": query,
                "expanded_query": expanded_query,
                "query_intent": query_info,
                "context_data": context_data,
                "processing_time": (datetime.now() - start_time).total_seconds()
            }

            # 根据请求包含搜索结果
            if return_search_results:
                response["search_results"] = search_results["results"]

            # 包含提示词
            if return_prompt:
                response["prompt"] = prompt

            return response

        except Exception as e:
            logger.error(f"处理查询失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        分析查询意图和要素

        Args:
            query: 用户查询

        Returns:
            查询分析结果
        """
        # 提取关键词
        keywords = extract_keywords(query, max_keywords=5)

        # 检测查询类型
        query_lower = query.lower()

        # 初始化查询信息
        query_info = {
            "keywords": keywords,
            "contains_time_reference": False,
            "contains_service_reference": False,
            "contains_error_reference": False,
            "query_type": "general"
        }

        # 检测时间引用
        time_patterns = [
            r'昨天', r'今天', r'上周', r'前天', r'最近',
            r'(\d+)\s*(分钟|小时|天|周|月)前',
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
            r'(\d{1,2}:\d{2})'
        ]

        for pattern in time_patterns:
            if re.search(pattern, query):
                query_info["contains_time_reference"] = True
                break

        # 检测服务引用
        service_patterns = [
            r'服务[^,，。；]*',
            r'模块[^,，。；]*',
            r'系统[^,，。；]*',
            r'应用[^,，。；]*'
        ]

        for pattern in service_patterns:
            service_match = re.search(pattern, query)
            if service_match:
                query_info["contains_service_reference"] = True
                query_info["possible_service"] = service_match.group(0)
                break

        # 检测错误引用
        error_patterns = [
            r'错误', r'异常', r'失败', r'问题', r'故障',
            r'error', r'exception', r'fail', r'issue', r'bug'
        ]

        for pattern in error_patterns:
            if pattern in query_lower:
                query_info["contains_error_reference"] = True
                break

        # 确定查询类型
        if any(term in query_lower for term in ['为什么', '原因', 'why', 'reason']):
            query_info["query_type"] = "cause_analysis"
        elif any(term in query_lower for term in ['怎么解决', '如何修复', '解决方案', 'how to fix', 'solution']):
            query_info["query_type"] = "solution_seeking"
        elif any(term in query_lower for term in ['多少', '数量', '频率', 'how many', 'count', 'frequency']):
            query_info["query_type"] = "statistical"
        elif query_info["contains_error_reference"]:
            query_info["query_type"] = "error_related"

        return query_info

    def _expand_query(self, query: str, query_info: Dict[str, Any]) -> str:
        """
        扩展查询以提高召回率

        Args:
            query: 原始查询
            query_info: 查询分析结果

        Returns:
            扩展后的查询
        """
        # 简单扩展策略
        expanded_terms = []

        # 根据查询类型添加相关术语
        if query_info["query_type"] == "error_related":
            if "error" not in query.lower() and "错误" not in query:
                expanded_terms.extend(["error", "exception", "错误", "异常"])

        if query_info["query_type"] == "cause_analysis":
            expanded_terms.extend(["cause", "reason", "原因"])

        if query_info["query_type"] == "solution_seeking":
            expanded_terms.extend(["solution", "fix", "resolve", "解决"])

        # 如果没有特殊扩展需求，直接返回原查询
        if not expanded_terms:
            return query

        # 合并原查询和扩展术语
        return f"{query} {' '.join(expanded_terms)}"

    def _build_context(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        从搜索结果构建上下文

        Args:
            search_results: 搜索结果列表

        Returns:
            上下文数据
        """
        if not search_results:
            return {"context": "", "sources": []}

        # 限制结果数量
        top_results = search_results[:self.max_context_logs]

        # 构建上下文
        context_parts = []
        sources = []

        for i, result in enumerate(top_results):
            # 提取结果数据
            timestamp = result.get("timestamp", "")
            log_level = result.get("log_level", "")
            message = result.get("message", "")
            service = result.get("service_name", "")
            template = result.get("template", "")
            cluster_id = result.get("cluster_id", "")

            # 构建上下文片段，包含日志信息和元数据
            context_part = (
                f"[日志 {i + 1}]\n"
                f"时间: {timestamp}\n"
                f"级别: {log_level}\n"
                f"服务: {service}\n"
                f"内容: {message}"
            )

            # 如果有模板信息，添加到上下文
            if template:
                context_part += f"\n模板: {template}"

            context_parts.append(context_part)

            # 记录来源信息
            sources.append({
                "id": result.get("id", ""),
                "timestamp": timestamp,
                "service": service,
                "level": log_level,
                "score": result.get("final_score", 0),
                "message_preview": message[:100] + ("..." if len(message) > 100 else "")
            })

        # 组合完整上下文
        full_context = "\n\n".join(context_parts)

        # 确保不超过令牌限制
        rough_token_count = len(full_context.split()) + sum(
            1.5 for char in full_context if '\u4e00' <= char <= '\u9fff')

        if rough_token_count > self.token_limit:
            # 简单截断，实际应用中可能需要更复杂的处理
            char_limit = int(self.token_limit * 0.6)  # 粗略转换为字符限制
            full_context = full_context[:char_limit] + "...(内容已截断)"

        return {
            "context": full_context,
            "sources": sources,
            "token_estimate": rough_token_count
        }

    def _generate_prompt(self, query: str, context: str) -> str:
        """
        为LLM生成提示词

        Args:
            query: 用户查询
            context: 上下文信息

        Returns:
            格式化的提示词
        """
        # 使用模板生成提示词
        prompt = self.prompt_template.format(
            context=context,
            query=query
        )

        return prompt

    def set_prompt_template(self, template: str) -> None:
        """
        设置提示词模板

        Args:
            template: 新的提示词模板
        """
        if not template:
            logger.warning("提供的模板为空，使用默认模板")
            self.prompt_template = self.default_prompt_template
            return

        # 验证模板是否包含必要的占位符
        if "{context}" not in template or "{query}" not in template:
            logger.warning("模板必须包含 {context} 和 {query} 占位符，使用默认模板")
            return

        self.prompt_template = template
        logger.info("已更新提示词模板")

    def get_default_prompt_template(self) -> str:
        """
        获取默认提示词模板

        Returns:
            默认提示词模板
        """
        return self.default_prompt_template

    # 修复 Dify Workflow 端点以包含 LLM 提示词

    def generate_dify_workflow_data(
            self,
            query: str,
            filters: Optional[Dict[str, Any]] = None,
            time_range: Optional[Any] = None,
            services: Optional[List[str]] = None,
            log_levels: Optional[List[str]] = None,
            include_vectors: bool = True,  # 是否包含向量搜索结果
            include_prompt: bool = True  # 新参数：是否包含 LLM 提示词
    ) -> Dict[str, Any]:
        """
        为 Dify 工作流生成数据，包括 LLM 提示词

        Args:
            query: 用户查询
            filters: 过滤条件
            time_range: 时间范围(开始时间, 结束时间) - 可以是各种格式
            services: 服务名列表
            log_levels: 日志级别列表
            include_vectors: 是否包含向量搜索结果
            include_prompt: 是否包含 LLM 提示词

        Returns:
            工作流数据包括搜索结果和 LLM 提示词
        """
        import logging
        logger = logging.getLogger(__name__)

        # 确保时间范围格式正确
        processed_time_range = None
        meta_time_range = {"start": None, "end": None}

        if time_range:
            logger.info(f"原始时间范围: {time_range}, 类型: {type(time_range)}")

            # 如果是元组或列表
            if isinstance(time_range, (tuple, list)) and len(time_range) >= 2:
                start_time, end_time = time_range[0], time_range[1]
                logger.info(f"解析时间范围: start={start_time}, end={end_time}")

                # 记录元数据中的时间范围（用于响应）
                meta_time_range["start"] = str(start_time)
                meta_time_range["end"] = str(end_time)

                # 保留原始时间范围格式
                processed_time_range = time_range
            else:
                logger.warning(f"时间范围格式不正确: {time_range}")

        try:
            # 执行混合搜索
            logger.info(f"执行混合搜索，include_vectors={include_vectors}")
            search_results = self.hybrid_search.search(
                query=query,
                filters=filters,
                include_vectors=include_vectors,  # 控制是否进行向量搜索
                time_range=processed_time_range,
                services=services,
                log_levels=log_levels,
            )

            # 构建上下文和生成 LLM 提示词
            context_data = {}
            prompt = None

            if include_prompt:
                # 构建搜索结果上下文
                context_data = self._build_context(search_results.get("results", []))

                # 生成提示词
                prompt = self._generate_prompt(query, context_data.get("context", ""))
                logger.info(f"已生成 LLM 提示词，长度: {len(prompt) if prompt else 0} 字符")

            # 准备工作流数据
            workflow_data = {
                "query": query,
                "search_results": search_results.get("results", []),
                "meta": {
                    "total": search_results.get("total", 0),
                    "es_results_count": search_results.get("es_results_count", 0),
                    "vector_results_count": search_results.get("vector_results_count", 0),
                    "time_range": meta_time_range
                }
            }

            # 添加提示词和上下文（如果有）
            if include_prompt and prompt:
                workflow_data["prompt"] = prompt
                workflow_data["context_data"] = context_data

            return workflow_data

        except Exception as e:
            logger.error(f"生成 Dify 工作流数据出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

            # 返回一个基本的响应，而不是引发异常
            return {
                "query": query,
                "search_results": [],
                "meta": {
                    "total": 0,
                    "error": str(e),
                    "time_range": meta_time_range
                }
            }

    def generate_time_parsing_prompt( self,query: str) -> str:
        """
        生成用于解析时间表达式的提示词

        Args:
            query: 用户查询文本

        Returns:
            用于LLM的提示词
        """
        prompt = f"""
        你是一个专门解析时间表达式的助手。你的任务是从用户查询中提取时间信息，并以JSON格式返回,不要携带除JSON以外任何内容。
        从以下查询中提取时间相关信息，并转换为具体的时间范围:

        查询: "{query}"

        请以JSON格式返回，包含以下字段:
        - has_time_reference: 布尔值，表示是否包含时间引用
        - start_time: 开始时间(ISO格式)
        - end_time: 结束时间(ISO格式)
        - time_type: "absolute"(具体时间点) 或 "relative"(相对时间)
        - original_expression: 原始时间表达式

        如果查询不包含时间引用，则has_time_reference为false，其余字段为null。

        当前日期是：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        示例1:
        查询: "今天有什么报错信息"
        返回:
        {{
        "has_time_reference": true,
          "start_time": "2025-04-13T00:00:00",
          "end_time": "2025-04-13T23:59:59",
          "time_type": "relative",
          "original_expression": "今天"
        }}

        示例2:
        查询: "14点到19点之间的错误日志"
        返回:
        {{
        "has_time_reference": true,
          "start_time": "2025-04-13T14:00:00",
          "end_time": "2025-04-13T19:00:00",
          "time_type": "absolute",
          "original_expression": "14点到19点之间"
        }}

        示例3:
        查询: "最近3小时的系统警告"
        返回:
         {{
        "has_time_reference": true,
          "start_time": "2025-04-13T11:30:00",
          "end_time": "2025-04-13T14:30:00",
          "time_type": "relative",
          "original_expression": "最近3小时"
        }}

        示例4:
        查询: "显示服务器错误日志"
        返回:
        {{
        "has_time_reference": false,
          "start_time": null,
          "end_time": null,
          "time_type": null,
          "original_expression": null
        }}
        """

        return prompt