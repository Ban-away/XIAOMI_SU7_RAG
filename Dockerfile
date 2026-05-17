# 基于 Python 3.10 官方镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV RAG_BASE_DIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 复制项目代码
COPY . .

# 创建必要目录
RUN mkdir -p data/{processed_docs,saved_index,qa_pairs,summary_data,rerank_data,saved_images,mongodb/{data,log}} \
    && mkdir -p log models

# 端口暴露
EXPOSE 6000 8000

# 启动命令（默认启动在线推理）
CMD ["python", "infer.py"]