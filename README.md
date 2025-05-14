# LogAI系统生产环境部署指南

本指南提供在生产环境中部署和使用LogAI系统的完整步骤，包括API调用顺序、关键参数和Dify工作流集成方案。LogAI系统专为C# ABP框架日志分析设计，结合向量检索和大语言模型提供智能日志分析能力。

## 第一阶段：系统初始化

### 步骤1：初始数据处理与向量化

首先需要处理初始日志数据并生成向量表示：

```bash
curl -X POST "http://your-server:8000/api/process_and_vectorize" -H "Content-Type: application/json" -d '
{
  "source_index": "your-abp-logs-index-*",  # 你的ABP日志索引模式
  "target_index": "logai-processed",        # 处理后的目标索引
  "time_range_start": "2025-04-09T00:00:00.000Z",  # 初始数据范围开始时间（通常是过去1个月）
  "time_range_end": "2025-05-09T23:59:59.999Z",    # 初始数据范围结束时间
  "max_logs": 100000,                       # 最大处理日志数量
  "prefilter": true,                        # 启用预筛选去重
  "time_window_minutes": 30                 # 时间窗口（分钟）
}'
```

**参数说明：**
- `source_index`：ABP日志的源索引模式
- `target_index`：处理后存储的索引名称（推荐使用'logai-processed'）
- `time_range_start/time_range_end`：初始数据范围（建议处理近1-3个月的日志）
- `max_logs`：初始处理的最大日志数量（根据系统资源调整）
- `prefilter`：是否启用预筛选去重（推荐启用）
- `time_window_minutes`：时间窗口，用于分组去重（默认30分钟）

### 步骤2：设置增量处理定时任务

在生产服务器上设置一个cron任务，定期执行增量处理：

```bash
# 创建向量化服务
cat > /etc/systemd/system/log-vectorizer.service << EOF
[Unit]
Description=Vectorize processed logs

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'curl -X POST "http://localhost:8000/api/process_and_vectorize" -H "Content-Type: application/json" -d \'{\"max_logs\": 5000, \"prefilter\": true, \"time_window_minutes\": 30}\''

[Install]
WantedBy=multi-user.target
EOF

# 创建向量化定时器
cat > /etc/systemd/system/log-vectorizer.timer << EOF
[Unit]
Description=Run log vectorizer every 10 minutes

[Timer]
OnBootSec=10min
OnUnitActiveSec=10min
AccuracySec=1s

[Install]
WantedBy=timers.target
EOF

# 启用并启动timer
systemctl daemon-reload
systemctl enable log-vectorizer.timer
systemctl start log-vectorizer.timer
```

**注意：** 使用 `process_and_vectorize` 端点而不是 `vectorize_logs` 端点，因为前者同时处理日志并生成向量表示。

## 第二阶段：配置Dify工作流

### 步骤1：创建基础查询工作流

在Dify中创建一个新工作流，用于处理日志查询。工作流应包含以下API调用：

**1. 查询分析：** 调用LogAI的查询分类API

```bash
curl -X POST "http://your-server:8000/api/classify_query" -H "Content-Type: application/json" -d '
{
  "query": "${user_query}"  # Dify变量：用户输入的查询
}'
```

**2. 更新权重：** 根据查询分析结果调整搜索权重

```bash
curl -X POST "http://your-server:8000/api/update_search_weights" -H "Content-Type: application/json" -d '
{
  "weights_config": ${query_analysis_result},  # Dify变量：上一步的分析结果
  "disable_dynamic_weights": true              # 禁用动态权重调整，使用固定权重
}'
```

**3. 执行日志搜索：** 使用dify_workflow端点获取相关日志

```bash
curl -X POST "http://your-server:8000/api/dify_workflow" -H "Content-Type: application/json" -d '
{
  "query": "${user_query}",             # Dify变量：用户输入的查询
  "time_range_start": "${search_start_time}",  # Dify变量：搜索开始时间
  "time_range_end": "${search_end_time}",      # Dify变量：搜索结束时间
  "services": ${selected_services},            # Dify变量：选择的服务（数组）
  "log_levels": ${selected_log_levels},        # Dify变量：选择的日志级别（数组）
  "include_vectors": true,                     # 启用向量搜索
  "include_prompt": true                       # 包含LLM提示词
}'
```

**4. 发送到LLM：** 将结果中的prompt发送给LLM进行处理

### 步骤2：创建高级分析工作流

为特定场景创建专门的工作流：

**错误分析工作流：**

```bash
curl -X POST "http://your-server:8000/api/analyze_anomalies" -H "Content-Type: application/json" -d '
{
  "start_time": "${search_start_time}",
  "end_time": "${search_end_time}",
  "max_logs": 10000,
  "services": ${selected_services}
}'
```

**模式分析工作流：**

```bash
curl -X POST "http://your-server:8000/api/analyze_patterns" -H "Content-Type: application/json" -d '
{
  "start_time": "${search_start_time}",
  "end_time": "${search_end_time}",
  "max_logs": 10000,
  "services": ${selected_services}
}'
```

## 第三阶段：Dify工作流详细配置

在Dify中设置工作流，创建以下关键步骤：

### 基本工作流：日志查询分析

**输入节点：**
- 用户查询（文本）
- 时间范围（开始/结束）
- 服务选择（多选）
- 日志级别（多选）

**处理节点：**
- 调用查询分类API
- 调用权重更新API
- 调用dify_workflow API
- 调用LLM处理结果

**输出节点：**
- 格式化的分析结果

### 完整Dify工作流示例

```json
{
  "workflow": {
    "name": "LogAI分析",
    "nodes": [
      {
        "id": "input",
        "type": "input",
        "fields": [
          {"name": "user_query", "type": "text", "label": "日志查询"},
          {"name": "search_start_time", "type": "datetime", "label": "开始时间"},
          {"name": "search_end_time", "type": "datetime", "label": "结束时间"},
          {"name": "selected_services", "type": "multi-select", "label": "服务", "options": []},
          {"name": "selected_log_levels", "type": "multi-select", "label": "日志级别", "options": []}
        ]
      },
      {
        "id": "classify_query",
        "type": "http_request",
        "url": "http://your-server:8000/api/classify_query",
        "method": "POST",
        "body": {
          "query": "${user_query}"
        },
        "output_variable": "query_analysis"
      },
      {
        "id": "update_weights",
        "type": "http_request",
        "url": "http://your-server:8000/api/update_search_weights",
        "method": "POST",
        "body": {
          "weights_config": "${query_analysis.classification}",
          "disable_dynamic_weights": true
        }
      },
      {
        "id": "search_logs",
        "type": "http_request",
        "url": "http://your-server:8000/api/dify_workflow",
        "method": "POST",
        "body": {
          "query": "${user_query}",
          "time_range_start": "${search_start_time}",
          "time_range_end": "${search_end_time}",
          "services": "${selected_services}",
          "log_levels": "${selected_log_levels}",
          "include_vectors": true,
          "include_prompt": true
        },
        "output_variable": "search_results"
      },
      {
        "id": "llm_analysis",
        "type": "llm",
        "prompt": "${search_results.prompt}",
        "output_variable": "final_analysis"
      },
      {
        "id": "output",
        "type": "output",
        "data": {
          "analysis": "${final_analysis}",
          "logs": "${search_results.search_results}"
        }
      }
    ]
  }
}
```

## 注意事项和最佳实践

### 索引管理
- 定期检查logai-processed索引大小并考虑索引生命周期管理
- 对于大型生产环境，考虑按月分片索引
- 使用 `/api/diagnose_indices` 端点检查索引状态

### 性能优化
- 调整max_logs参数以平衡处理速度和系统负载
- 根据日志量级调整增量处理频率
- 配置去重参数控制数据量

### 搜索权重调整
- 针对不同类型查询设置合适的ES与向量搜索权重
- 对于错误查询，建议设置更高的ES权重（0.7 ES, 0.3 向量）
- 对于语义问题，建议设置更高的向量权重（0.3 ES, 0.7 向量）
- 使用 `disable_dynamic_weights: true` 参数确保权重设置生效

### 向量搜索调试
- 如果向量搜索无法返回结果，可以使用 `check_qdrant.py` 工具诊断问题
- 确保 Qdrant 集合中有数据，检查数据时间范围是否匹配
- 对于中文查询，确认向量模型支持中文嵌入

### 监控和警报
- 监控LogAI服务的健康状态（使用/health端点）
- 设置处理失败的告警机制
- 定期检查日志向量化是否正常进行

### 权限控制
- 为Dify工作流和LogAI服务间的通信设置适当的身份验证
- 限制对敏感API的访问

### 用户界面整合
- 考虑创建专门的日志分析界面，集成Dify工作流
- 提供时间和服务选择的可视化界面

按照上述步骤配置，您将拥有一个完整的、自动化的日志分析系统，能够处理C# ABP框架的日志数据，并利用AI提供智能分析和洞察。系统结合了基于关键词的ES搜索和语义向量搜索，可以根据不同查询类型自动或手动调整搜索权重，提供最佳结果。