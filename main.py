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
        es_connector=es_connector,
        vector_store=vector_store,
        embedding_model=embedding_model,
        es_weight=hybrid_search_config.get("es_weight", 0.3),
        vector_weight=hybrid_search_config.get("vector_weight", 0.7),
        final_results_limit=hybrid_search_config.get("top_k", 20)
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
def incremental_process(background_tasks: BackgroundTasks, components: Dict = Depends(get_components)):
    """
    启动增量处理
    """
    # 使用后台任务，避免长时间运行的请求
    background_tasks.add_task(
        components["log_processor"].process_incremental,
        time_window=timedelta(hours=1),
        max_logs=10000,
        save_to_file=True,
        save_to_es=True
    )

    return {"status": "success", "message": "增量处理任务已启动"}


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