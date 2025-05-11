# main.py
import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Query, Body, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# 导入自定义模块
from config import get_config, get_model_config
from utils import setup_logging, get_logger
from data.es_connector import ESConnector
from data.vector_store import QdrantVectorStore
from models.embedding import LocalSentenceTransformerEmbedding
from processing.preprocessor import LogPreprocessor
from processing.extractor import LogProcessor
from processing.vectorizer import LogVectorizer
from retrieval.hybrid_search import HybridSearch
from retrieval.query_processor import QueryProcessor
from analysis.anomaly_detector import LogAnomalyDetector
from analysis.pattern_analyzer import LogPatternAnalyzer

# 初始化日志
logger = get_logger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="日志AI分析系统",
    description="基于LLM和向量检索的日志智能分析系统",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置为特定的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 定义Pydantic模型，用于请求和响应
class SearchQuery(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    services: Optional[List[str]] = None
    log_levels: Optional[List[str]] = None
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None
    include_vectors: bool = True
    max_results: int = 20


class PromptQuery(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    services: Optional[List[str]] = None
    log_levels: Optional[List[str]] = None
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None
    return_search_results: bool = False


class DifyWorkflowQuery(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    services: Optional[List[str]] = None
    log_levels: Optional[List[str]] = None
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None


class ProcessLogsRequest(BaseModel):
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    max_logs: int = 10000
    query_filter: Optional[Dict[str, Any]] = None
    save_to_file: bool = True
    save_to_es: bool = True
    target_index: Optional[str] = None


class VectorizeLogsRequest(BaseModel):
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    max_logs: int = 10000
    query_filter: Optional[Dict[str, Any]] = None
    content_column: str = "message"
    service_name_column: str = "service_name"


class AnalysisRequest(BaseModel):
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    max_logs: int = 10000
    query_filter: Optional[Dict[str, Any]] = None
    services: Optional[List[str]] = None
    log_levels: Optional[List[str]] = None


# 请求和响应模型
class QueryClassificationRequest(BaseModel):
    query: str
    context: Optional[str] = None

class QueryClassificationResponse(BaseModel):
    success: bool
    query: str
    system_prompt: str
    user_prompt: str

# 去重配置请求模型
class DeduplicationConfigRequest(BaseModel):
    enabled: bool = True
    similarity_threshold: Optional[float] = None
    novelty_threshold: Optional[float] = None
    time_window_hours: Optional[int] = None
    min_cluster_size: Optional[int] = None
    keep_latest_count: Optional[int] = None
    keep_anomalies: Optional[bool] = None

class WeightUpdateRequest(BaseModel):
    es_weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="ES搜索权重")
    vector_weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="向量搜索权重")
    weights_config: Optional[Union[Dict[str, Any], str]] = Field(None,
                                                                 description="LLM分析的权重配置，可以是JSON字符串或字典")


# 全局组件实例
es_connector = None
vector_store = None
embedding_model = None
log_processor = None
log_vectorizer = None
hybrid_search = None
query_processor = None
anomaly_detector = None
pattern_analyzer = None


# 初始化全局组件
def initialize_components():
    global es_connector, vector_store, embedding_model, log_processor, log_vectorizer
    global hybrid_search, query_processor, anomaly_detector, pattern_analyzer

    # 获取配置
    config = get_config()
    model_config = get_model_config()

    # 初始化日志
    log_level = config.get("app.log_level", "INFO")
    log_dir = config.get("app.log_dir", "./logs")
    app_name = config.get("app.name", "log-ai-analyzer")
    setup_logging(log_level, log_dir, app_name)

    # 初始化ES连接器
    es_config = config.get("elasticsearch", {})
    es_connector = ESConnector(
        es_host=es_config.get("host", "localhost"),
        es_port=es_config.get("port", 9200),
        es_user=es_config.get("username"),
        es_password=es_config.get("password"),
        es_index_pattern=es_config.get("index_pattern", "logstash-*"),
        use_ssl=es_config.get("use_ssl", False),
        verify_certs=es_config.get("verify_certs", False)
    )

    # 初始化目标ES连接器（带IK分词器）
    target_es_config = config.get("target_elasticsearch", {})
    target_es_connector = ESConnector(
        es_host=target_es_config.get("host", "localhost"),
        es_port=target_es_config.get("port", 9200),
        es_user=target_es_config.get("username"),
        es_password=target_es_config.get("password"),
        es_index_pattern=target_es_config.get("index", "logai-processed"),  # 注意这里使用index而非index_pattern
        use_ssl=target_es_config.get("use_ssl", False),
        verify_certs=target_es_config.get("verify_certs", False)
    )

    # 初始化嵌入模型
    embedding_config = model_config.get("embedding", {})
    embedding_model = LocalSentenceTransformerEmbedding(
        model_name=embedding_config.get("model_name", "bge-m3"),
        model_path=embedding_config.get("model_path", "/models/sentence-transformers"),
        batch_size=embedding_config.get("batch_size", 32),
        max_seq_length=embedding_config.get("max_seq_length", 512),
        device=embedding_config.get("device", "cpu")
    )

    # 初始化向量存储
    qdrant_config = config.get("qdrant", {})
    vector_store = QdrantVectorStore(
        host=qdrant_config.get("host", "localhost"),
        port=qdrant_config.get("port", 6333),
        grpc_port=qdrant_config.get("grpc_port", 6334),
        api_key=qdrant_config.get("api_key"),
        prefer_grpc=qdrant_config.get("prefer_grpc", True),
        timeout=qdrant_config.get("timeout", 60.0),
        collection_name=qdrant_config.get("collection_name", "log_vectors"),
        vector_size=embedding_model.get_dimension()
    )

    # 确保集合存在
    vector_store.ensure_collection_exists()

    # 初始化日志处理器
    log_processor = LogProcessor(
        es_connector=es_connector,
        target_es_connector=target_es_connector,  # 需要修改LogProcessor类接受这个参数
        output_dir=config.get("app.output_dir", "./output")
    )

    # 初始化日志向量化处理器
    log_vectorizer = LogVectorizer(
        embedding_model=embedding_model,
        vector_store=vector_store
    )

    # 初始化混合搜索
    retrieval_config = config.get("retrieval", {})
    hybrid_search_config = retrieval_config.get("hybrid_search", {})
    hybrid_search = HybridSearch(
        es_connector=target_es_connector,
        vector_store=vector_store,
        embedding_model=embedding_model,
        es_weight=hybrid_search_config.get("es_weight", 0.3),
        vector_weight=hybrid_search_config.get("vector_weight", 0.7),
        final_results_limit=hybrid_search_config.get("top_k", 20),
        rerank_url = hybrid_search_config.get("rerank_url","http://localhost:8091/rerank")  # 您的重排序服务URL
    )

    # 初始化查询处理器
    query_processor = QueryProcessor(
        hybrid_search=hybrid_search,
        max_context_logs=5,
        token_limit=4000
    )

    # 初始化异常检测器
    anomaly_detector = LogAnomalyDetector(
        time_window=timedelta(hours=1),
        sensitivity=0.95
    )

    # 初始化模式分析器
    pattern_analyzer = LogPatternAnalyzer(
        time_window=timedelta(minutes=5),
        min_confidence=0.6
    )

    # 初始化日志处理器时传入去重配置
    log_processor = LogProcessor(
        es_connector=es_connector,
        target_es_connector=target_es_connector,
        output_dir=config.get("app.output_dir", "./output")
    )

    # 从配置中读取去重设置
    dedup_config = config.get("deduplication", {})
    if dedup_config:
        from deduplication import LogDeduplicator
        deduplicator = LogDeduplicator(
            similarity_threshold=dedup_config.get("similarity_threshold", 0.92),
            time_window_hours=dedup_config.get("time_window_hours", 2),
            min_cluster_size=dedup_config.get("min_cluster_size", 2),
            novelty_threshold=dedup_config.get("novelty_threshold", 0.05),
            keep_latest_count=dedup_config.get("keep_latest_count", 3),
            keep_anomalies=dedup_config.get("keep_anomalies", True)
        )
        log_processor.deduplicator = deduplicator

    logger.info("所有组件初始化完成")


# 依赖项，确保组件已初始化
def get_components():
    if es_connector is None:
        initialize_components()
    return {
        "es_connector": es_connector,
        "vector_store": vector_store,
        "embedding_model": embedding_model,
        "log_processor": log_processor,
        "log_vectorizer": log_vectorizer,
        "hybrid_search": hybrid_search,
        "query_processor": query_processor,
        "anomaly_detector": anomaly_detector,
        "pattern_analyzer": pattern_analyzer
    }


# API路由

@app.get("/")
def read_root():
    return {"status": "ok", "message": "日志AI分析系统API服务正常运行"}


@app.get("/health")
def health_check():
    # 简单的健康检查
    components = get_components()
    es_health = True
    try:
        components["es_connector"].es_client.ping()
    except:
        es_health = False

    return {
        "status": "healthy",
        "components": {
            "elasticsearch": es_health,
            "vector_store": components["vector_store"] is not None,
            "embedding_model": components["embedding_model"] is not None
        },
        "timestamp": datetime.now().isoformat()
    }





@app.get("/api/deduplication/stats")
def get_deduplication_stats(components: Dict = Depends(get_components)):
    """获取日志去重统计信息"""
    deduplicator = components["log_processor"].deduplicator
    return {
        "status": "success",
        "stats": deduplicator.get_stats()
    }


@app.post("/api/deduplication/config")
def configure_deduplication(
        request: DeduplicationConfigRequest,
        components: Dict = Depends(get_components)
):
    """配置日志去重系统"""
    deduplicator = components["log_processor"].deduplicator

    # 更新配置参数
    if request.similarity_threshold is not None:
        deduplicator.similarity_threshold = request.similarity_threshold

    if request.novelty_threshold is not None:
        deduplicator.novelty_threshold = request.novelty_threshold

    if request.time_window_hours is not None:
        deduplicator.time_window_hours = request.time_window_hours

    if request.min_cluster_size is not None:
        deduplicator.min_cluster_size = request.min_cluster_size

    if request.keep_latest_count is not None:
        deduplicator.keep_latest_count = request.keep_latest_count

    if request.keep_anomalies is not None:
        deduplicator.keep_anomalies = request.keep_anomalies

    return {
        "status": "success",
        "message": "去重配置已更新",
        "current_config": {
            "similarity_threshold": deduplicator.similarity_threshold,
            "novelty_threshold": deduplicator.novelty_threshold,
            "time_window_hours": deduplicator.time_window_hours,
            "min_cluster_size": deduplicator.min_cluster_size,
            "keep_latest_count": deduplicator.keep_latest_count,
            "keep_anomalies": deduplicator.keep_anomalies
        }
    }


@app.post("/api/deduplication/reset")
def reset_deduplication_stats(components: Dict = Depends(get_components)):
    """重置去重统计信息"""
    deduplicator = components["log_processor"].deduplicator
    deduplicator.reset_stats()
    return {
        "status": "success",
        "message": "去重统计已重置"
    }

@app.post("/api/search")
def search_logs(query: SearchQuery, components: Dict = Depends(get_components)):
    """
    搜索日志
    """
    # 准备时间范围
    time_range = None
    if query.time_range_start and query.time_range_end:
        time_range = (query.time_range_start, query.time_range_end)

    # 执行搜索
    search_results = components["hybrid_search"].search(
        query=query.query,
        filters=query.filters,
        include_vectors=query.include_vectors,
        time_range=time_range,
        services=query.services,
        log_levels=query.log_levels,
    )

    # 限制结果数量
    if "results" in search_results:
        search_results["results"] = search_results["results"][:query.max_results]

    return search_results


@app.post("/api/generate_prompt")
def generate_prompt(query: PromptQuery, components: Dict = Depends(get_components)):
    """
    生成LLM提示词
    """
    # 准备时间范围
    time_range = None
    if query.time_range_start and query.time_range_end:
        time_range = (query.time_range_start, query.time_range_end)

    # 处理查询
    result = components["query_processor"].process_query(
        query=query.query,
        filters=query.filters,
        return_search_results=query.return_search_results,
        return_prompt=True,
        time_range=time_range,
        services=query.services,
        log_levels=query.log_levels
    )

    return result

# 定义请求模型
class QueryRequest(BaseModel):
    query: str
@app.post('/api/parse_time')
def parse_time_api(request: QueryRequest, components: Dict = Depends(get_components)):
    """
    解析时间信息
    """
    # 使用 `components["query_processor"]` 调用方法
    template = components["query_processor"].generate_time_parsing_prompt(request.query)

    # 返回生成的模板（prompt）
    return {"prompt": template}


@app.post("/api/classify_query")
def classify_query_endpoint(request: QueryClassificationRequest) -> QueryClassificationResponse:
    """提供查询分类所需的提示词和内容，专为C# ABP框架日志优化"""

    query = request.query

    # 为C# ABP框架日志优化的系统提示词
    system_prompt = """
    你是一个专业的日志查询分析器，专注于C# ABP框架的日志分析。你的任务是分析用户的查询意图，并确定在混合检索系统中使用的最佳权重。

    请分析查询的语义和意图，然后确定它属于以下哪种类型：
    1. 错误定位型查询：寻找特定错误、异常、堆栈跟踪或错误代码的查询
    2. 性能分析型查询：与响应时间、延迟、超时或性能瓶颈相关的查询
    3. 安全审计型查询：寻找授权失败、身份验证问题或安全警告的查询
    4. 流程跟踪型查询：跟踪特定请求、事务或用户会话流程的查询
    5. 依赖服务型查询：与外部服务、API调用或数据库交互相关的查询
    6. 通用型查询：不属于以上类别的一般性查询

    对于不同类型的查询，应分配不同的检索权重：
    - 错误定位型查询：词汇匹配(0.7)更重要，因为异常名称和错误消息有特定格式
    - 性能分析型查询：词汇匹配(0.4)和语义匹配(0.6)结合，因为性能问题描述多样
    - 安全审计型查询：词汇匹配(0.6)更重要，安全日志通常有固定模式
    - 流程跟踪型查询：语义匹配(0.7)更重要，因为需要理解整体流程上下文
    - 依赖服务型查询：词汇匹配(0.5)和语义匹配(0.5)平衡，涉及特定服务名和概念
    - 通用型查询：语义匹配(0.7)更重要，词汇匹配(0.3)次之

    请分析查询并返回JSON格式的结果，不要携带任何MarkDown格式形式，包含以下字段：
    - query_type: 查询类型(error_locating, performance, security_audit, process_tracing, dependency_service, general)
    - vector_weight: 语义匹配权重(0.0-1.0)
    - lexical_weight: 词汇匹配权重(0.0-1.0)
    - reasoning: 你的分析理由(简短说明)
    """

    # 构建用户提示词
    user_prompt = f"""
    请分析以下用户在C# ABP框架日志系统中的查询，确定查询类型并推荐合适的检索权重：

    用户查询: "{query}"

    请以JSON格式返回结果。只返回JSON对象，不要有其他说明文字。
    """

    # 返回提示词和查询
    return QueryClassificationResponse(
        success=True,
        query=query,
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )


@app.post("/api/update_search_weights")
def update_search_weights(
        request: WeightUpdateRequest,
        components: Dict = Depends(get_components)
):
    """更新混合搜索权重，支持直接指定权重或使用LLM分析结果"""
    import json

    hybrid_search = components["hybrid_search"]

    # 记录原始权重
    old_weights = {
        "es_weight": hybrid_search.es_weight,
        "vector_weight": hybrid_search.vector_weight
    }

    # 确定新权重
    new_es_weight = old_weights["es_weight"]
    new_vector_weight = old_weights["vector_weight"]
    reasoning = ""
    query_type = ""

    # 优先使用直接指定的权重
    if request.es_weight is not None:
        new_es_weight = request.es_weight

    if request.vector_weight is not None:
        new_vector_weight = request.vector_weight

    # 如果提供了LLM分析的权重配置
    weights_config = request.weights_config
    if weights_config:
        try:
            # 如果是字符串(JSON文本)，尝试解析成字典
            if isinstance(weights_config, str):
                weights_config = json.loads(weights_config)

            # 从LLM分析结果中提取权重
            vector_weight = float(weights_config.get('vector_weight', new_vector_weight))
            lexical_weight = float(weights_config.get('lexical_weight', new_es_weight))
            query_type = weights_config.get('query_type', 'general')
            reasoning = weights_config.get('reasoning', '')

            # 使用LLM分析的权重
            new_vector_weight = vector_weight
            new_es_weight = lexical_weight

            logger.info(
                f"使用LLM分析的查询权重: vector={vector_weight}, lexical={lexical_weight}, type={query_type}"
            )
        except Exception as e:
            logger.error(f"解析LLM权重配置时出错: {str(e)}")
            return {
                "status": "error",
                "message": f"解析权重配置失败: {str(e)}",
                "old_weights": old_weights
            }

    # 更新权重
    hybrid_search.set_weights(
        es_weight=new_es_weight,
        vector_weight=new_vector_weight
    )

    # 构建响应
    response = {
        "status": "success",
        "message": "搜索权重已更新",
        "old_weights": old_weights,
        "new_weights": {
            "es_weight": hybrid_search.es_weight,
            "vector_weight": hybrid_search.vector_weight
        }
    }

    # 如果使用了LLM分析，添加相关信息
    if query_type or reasoning:
        response["analysis"] = {
            "query_type": query_type,
            "reasoning": reasoning,
            "source": "llm_analysis"
        }

    return response

@app.post("/api/dify_workflow")
def generate_dify_workflow_data(query: DifyWorkflowQuery, components: Dict = Depends(get_components)):
    """
    生成用于Dify工作流的数据
    """
    # 准备时间范围
    time_range = None
    if query.time_range_start and query.time_range_end:
        time_range = (query.time_range_start, query.time_range_end)

    # 处理查询
    result = components["query_processor"].generate_dify_workflow_data(
        query=query.query,
        filters=query.filters,
        time_range=time_range,
        services=query.services,
        log_levels=query.log_levels
    )

    return result


@app.post("/api/process_logs")
def process_logs(request: ProcessLogsRequest, components: Dict = Depends(get_components)):
    """
    处理日志数据
    """
    # 执行处理
    processed_df, summary = components["log_processor"].extract_and_process(
        start_time=request.start_time,
        end_time=request.end_time,
        max_logs=request.max_logs,
        query_filter=request.query_filter,
        save_to_file=request.save_to_file,
        save_to_es=request.save_to_es,
        target_index=request.target_index
    )

    # 返回处理摘要
    return {
        "status": "success",
        "summary": summary,
        "record_count": len(processed_df) if processed_df is not None else 0
    }


@app.post("/api/vectorize_logs")
def vectorize_logs(request: VectorizeLogsRequest, components: Dict = Depends(get_components)):
    """
    向量化日志数据
    """
    # 首先提取并处理日志
    processed_df, _ = components["log_processor"].extract_and_process(
        start_time=request.start_time,
        end_time=request.end_time,
        max_logs=request.max_logs,
        query_filter=request.query_filter,
        save_to_file=False,
        save_to_es=False
    )

    if processed_df is None or processed_df.empty:
        return {"status": "error", "message": "没有日志数据可处理"}

    # 向量化处理后的数据
    total, successful = components["log_vectorizer"].process_dataframe(
        df=processed_df,
        content_column=request.content_column,
        service_name_column=request.service_name_column
    )

    return {
        "status": "success",
        "total_records": total,
        "successful_records": successful,
        "message": f"成功向量化 {successful}/{total} 条日志记录"
    }


@app.post("/api/analyze_anomalies")
def analyze_anomalies(request: AnalysisRequest, components: Dict = Depends(get_components)):
    """
    分析日志异常
    """
    # 首先提取日志
    df = components["es_connector"].extract_logs(
        start_time=request.start_time,
        end_time=request.end_time,
        max_logs=request.max_logs,
        query_filter=request.query_filter
    )

    if df.empty:
        return {"status": "error", "message": "没有日志数据可分析"}

    # 分析异常
    anomalies = components["anomaly_detector"].detect_anomalies(
        df=df,
        services=request.services,
        log_levels=request.log_levels
    )

    return anomalies


@app.post("/api/analyze_patterns")
def analyze_patterns(request: AnalysisRequest, components: Dict = Depends(get_components)):
    """
    分析日志模式
    """
    # 首先提取日志
    df = components["es_connector"].extract_logs(
        start_time=request.start_time,
        end_time=request.end_time,
        max_logs=request.max_logs,
        query_filter=request.query_filter
    )

    if df.empty:
        return {"status": "error", "message": "没有日志数据可分析"}

    # 分析模式
    patterns = components["pattern_analyzer"].analyze_patterns(
        df=df,
        services=request.services,
        log_levels=request.log_levels
    )

    return patterns


@app.post("/api/incremental_process")
def incremental_process(
        background_tasks: BackgroundTasks,
        components: Dict = Depends(get_components),
        request: Optional[Dict[str, Any]] = None,
):
    """
    启动增量处理
    """
    if request is None:
        request = {}

    # 获取配置
    config = get_config()

    # 使用请求参数或配置值
    source_index = request.get("source_index", config.get("elasticsearch.index_pattern"))
    target_index = request.get("target_index", config.get("target_elasticsearch.index"))

    # 使用后台任务，避免长时间运行的请求
    background_tasks.add_task(
        components["log_processor"].process_incremental,
        time_window=timedelta(hours=request.get("hours", 1)),
        max_logs=request.get("max_logs", 10000),
        save_to_file=request.get("save_to_file", True),
        save_to_es=request.get("save_to_es", True),
        source_index_pattern=source_index,
        target_index=target_index
    )

    return {
        "status": "success",
        "message": f"增量处理任务已启动，源索引: {source_index}, 目标索引: {target_index}"
    }

@app.post("/api/update_prompt_template")
def update_prompt_template(template: str = Body(..., embed=True), components: Dict = Depends(get_components)):
    """
    更新提示词模板
    """
    components["query_processor"].set_prompt_template(template)
    return {"status": "success", "message": "提示词模板已更新"}


@app.get("/api/prompt_template")
def get_prompt_template(components: Dict = Depends(get_components)):
    """
    获取当前提示词模板
    """
    template = components["query_processor"].prompt_template
    default_template = components["query_processor"].get_default_prompt_template()

    return {
        "current_template": template,
        "default_template": default_template,
        "is_default": template == default_template
    }


@app.get("/api/services")
def get_services(components: Dict = Depends(get_components)):
    """
    获取可用的服务列表
    """
    try:
        # 查询ES中的服务名称
        query = {
            "size": 0,
            "aggs": {
                "services": {
                    "terms": {
                        "field": "service_name",
                        "size": 100
                    }
                }
            }
        }

        result = components["es_connector"].es_client.search(
            index=components["es_connector"].es_index_pattern,
            body=query
        )

        services = [bucket["key"] for bucket in result["aggregations"]["services"]["buckets"]]

        return {
            "status": "success",
            "services": services,
            "count": len(services)
        }
    except Exception as e:
        logger.error(f"获取服务列表失败: {str(e)}")
        return {"status": "error", "message": str(e), "services": []}


@app.get("/api/log_levels")
def get_log_levels(components: Dict = Depends(get_components)):
    """
    获取可用的日志级别列表
    """
    try:
        # 查询ES中的日志级别
        query = {
            "size": 0,
            "aggs": {
                "log_levels": {
                    "terms": {
                        "field": "log_level",
                        "size": 20
                    }
                }
            }
        }

        result = components["es_connector"].es_client.search(
            index=components["es_connector"].es_index_pattern,
            body=query
        )

        log_levels = [bucket["key"] for bucket in result["aggregations"]["log_levels"]["buckets"]]

        return {
            "status": "success",
            "log_levels": log_levels,
            "count": len(log_levels)
        }
    except Exception as e:
        logger.error(f"获取日志级别列表失败: {str(e)}")
        return {"status": "error", "message": str(e), "log_levels": []}


@app.get("/api/stats")
def get_stats(components: Dict = Depends(get_components)):
    """
    获取系统统计信息
    """
    try:
        # 获取日志总数
        total_query = {
            "query": {"match_all": {}},
            "size": 0
        }

        total_result = components["es_connector"].es_client.count(
            index=components["es_connector"].es_index_pattern,
            body=total_query
        )

        total_logs = total_result["count"]

        # 获取错误日志数量
        error_query = {
            "query": {
                "terms": {
                    "log_level": ["ERROR", "FATAL"]
                }
            },
            "size": 0
        }

        error_result = components["es_connector"].es_client.count(
            index=components["es_connector"].es_index_pattern,
            body=error_query
        )

        error_logs = error_result["count"]

        # 向量化统计
        vector_stats = {
            "collection": components["vector_store"].collection_name,
            "vector_size": components["vector_store"].vector_size
        }

        # 尝试获取向量计数
        try:
            vector_count = components["vector_store"].client.count(
                collection_name=components["vector_store"].collection_name
            )
            vector_stats["count"] = vector_count.count
        except:
            vector_stats["count"] = "未知"

        return {
            "status": "success",
            "total_logs": total_logs,
            "error_logs": error_logs,
            "error_ratio": float(error_logs) / total_logs if total_logs > 0 else 0,
            "vectorization": vector_stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"获取统计信息失败: {str(e)}")
        return {"status": "error", "message": str(e)}


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"全局异常: {str(exc)}")
    import traceback
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": str(exc)}
    )


@app.post("/api/process_and_vectorize")
def process_and_vectorize(background_tasks: BackgroundTasks, components: Dict = Depends(get_components),request: Optional[Dict[str, Any]] = None):
    """
    处理并向量化日志数据
    """
    if request is None:
        request = {}

        # 获取配置
    config = get_config()

    # 使用请求参数或配置值
    source_index = request.get("source_index", config.get("elasticsearch.index_pattern"))
    target_index = request.get("target_index", config.get("target_elasticsearch.index"))

    # 使用后台任务，避免长时间运行的请求
    def combined_task():
        # 先处理日志
        processed_df, summary = components["log_processor"].process_incremental(
            time_window=timedelta(hours=1),
            max_logs=10000,
            save_to_file=True,
            save_to_es=True,
        source_index_pattern=source_index,
        target_index=target_index
        )
        # 再向量化
        if processed_df is not None and not processed_df.empty:
            components["log_vectorizer"].process_dataframe(processed_df)

        # 保存时间戳
        components["log_processor"].save_last_timestamp()

    background_tasks.add_task(combined_task)

    return {"status": "success", "message": "增量处理和向量化任务已启动"}

# 启动服务
if __name__ == "__main__":
    # 初始化组件
    initialize_components()

    # 获取服务配置
    config = get_config()
    host = config.get("app.host", "0.0.0.0")
    port = config.get("app.port", 8000)

    # 启动服务
    uvicorn.run(app, host=host, port=port)