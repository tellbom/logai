# logai

## /api/process_and_vectorize
* 用于进行从上个时间点从ES日志数据库到ES-IK数据库中
* 并且执行向量化后到Qdrant中
## /api/dify_workflow
* 用于混合检索后得到提示词与检索结果后续直接调用到dify工作流中的LLM进行问答结果

## /api/dify_workflow 调用堆栈顺序
* 客户端调用 /api/dify_workflow 端点，提供查询和过滤条件
* FastAPI处理请求，调用generate_dify_workflow_data函数
* QueryProcessor.generate_dify_workflow_data 方法处理请求
* 调用process_query方法处理查询
* 分析查询意图（_analyze_query）
* 扩展查询（_expand_query）
* 调用HybridSearch.search执行混合搜索
* 构建上下文（_build_context）
* 生成提示词（_generate_prompt）
准备Dify工作流数据
返回结果 给客户端，包含提示词和上下文

## /api/dify_workflow 调用堆栈顺序
下面是 /api/dify_workflow 的详细调用堆栈顺序，便于您后期调试：

main.py - @app.post("/api/dify_workflow") 路由

接收并解析 HTTP POST 请求
调用 components["query_processor"].generate_dify_workflow_data()


QueryProcessor.generate_dify_workflow_data() (在 retrieval/query_processor.py)

处理查询参数（时间范围、过滤条件等）
调用 self.process_query()


QueryProcessor.process_query() (在 retrieval/query_processor.py)

调用 self._analyze_query() 分析查询意图
调用 self._expand_query() 扩展查询
调用 self.hybrid_search.search() 执行混合搜索


HybridSearch.search() (在 retrieval/hybrid_search.py)

准备过滤条件 (_prepare_es_filters(), _prepare_vector_filters())
执行ES搜索 (_search_es())
执行向量搜索 (_search_vectors())
合并初步结果 (_merge_initial_results())
调用重排序 (_rerank_results()) -> 发送HTTP请求到重排序服务
最终合并结果 (_merge_final_results())


回到 QueryProcessor.process_query()

根据搜索结果调用 self._build_context() 构建上下文
调用 self._generate_prompt() 生成提示词


回到 QueryProcessor.generate_dify_workflow_data()

构建完整的 workflow_data 格式
返回结果给路由处理器


回到 main.py 路由处理器

返回最终 JSON 响应给客户端
