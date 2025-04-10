FROM python:3.10

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
# 设置 pip 使用阿里云的镜像源
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
COPY requirements.txt .
RUN pip install -r requirements.txt

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SENTENCE_TRANSFORMERS_HOME=/models \
    TZ=Asia/Shanghai

# 创建必要的目录
RUN mkdir -p ${SENTENCE_TRANSFORMERS_HOME}/sentence-transformers_bge-m3
COPY ./bge-m3  ${SENTENCE_TRANSFORMERS_HOME}/sentence-transformers_bge-m3

# 暴露API端口
EXPOSE 8000

# 容器启动后，您需要将源码挂载到/app目录

# 设置启动命令
CMD ["python", "main.py"]
