#!/bin/bash

# 当前时间
current_time=$(date -u +"%Y-%m-%dT%H:%M:%S.000Z")
# 一小时前
hour_ago=$(date -u -d "1 hour ago" +"%Y-%m-%dT%H:%M:%S.000Z")
# 12小时前
twelve_hours_ago=$(date -u -d "12 hours ago" +"%Y-%m-%dT%H:%M:%S.000Z")

# 添加当前时间的 INFO 日志
curl -X POST "http://localhost:9200/abp-logs-test/_doc" -H "Content-Type: application/json" -d "{
  \"@timestamp\": \"$current_time\",
  \"message\": \"Application started successfully\",
  \"log_level\": \"INFO\",
  \"service_name\": \"API.Host\"
}"

# 添加一小时前的 ERROR 日志
curl -X POST "http://localhost:9200/abp-logs-test/_doc" -H "Content-Type: application/json" -d "{
  \"@timestamp\": \"$hour_ago\",
  \"message\": \"Failed to connect to database: Connection timeout\",
  \"log_level\": \"ERROR\",
  \"service_name\": \"DataAccess\"
}"

# 添加 12 小时前的 WARNING 日志
curl -X POST "http://localhost:9200/abp-logs-test/_doc" -H "Content-Type: application/json" -d "{
  \"@timestamp\": \"$twelve_hours_ago\",
  \"message\": \"High memory usage detected: 85%\",
  \"log_level\": \"WARN\",
  \"service_name\": \"ResourceMonitor\"
}"

# 添加更多测试日志（随机生成）
for i in {1..10}; do
  random_hours=$((RANDOM % 24))
  random_time=$(date -u -d "$random_hours hours ago" +"%Y-%m-%dT%H:%M:%S.000Z")

  levels=("INFO" "WARN" "ERROR" "DEBUG")
  random_level=${levels[$((RANDOM % 4))]}

  services=("API.Host" "DataAccess" "Authentication" "BusinessLogic" "ResourceMonitor")
  random_service=${services[$((RANDOM % 5))]}

  messages=(
    "User login successful for userId: USER$i"
    "Processing request for endpoint: /api/data/$i"
    "Database query executed in 125ms"
    "Cache miss for key: CACHE_KEY_$i"
    "Memory usage increased to $((50 + RANDOM % 40))%"
    "API call to external service completed"
    "Exception occurred: System.NullReferenceException for object XYZ$i"
    "Background task $i completed successfully"
    "Configuration reloaded from source"
    "Rate limit reached for IP: 192.168.0.$i"
  )
  random_message=${messages[$((RANDOM % 10))]}

  curl -X POST "http://localhost:9200/abp-logs-test/_doc" -H "Content-Type: application/json" -d "{
    \"@timestamp\": \"$random_time\",
    \"message\": \"$random_message\",
    \"log_level\": \"$random_level\",
    \"service_name\": \"$random_service\"
  }"
done

# 刷新索引以确保数据立即可见
curl -X POST "http://localhost:9200/abp-logs-test/_refresh"

