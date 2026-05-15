# RAG Chunk Strategy Experiment Platform

一个用于对比不同文本切分策略对 RAG 系统检索质量影响的命令行实验平台。

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-336791.svg)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 项目简介

本项目提供了一个 CLI 实验平台，用于对比不同 chunk 策略对 RAG 检索和回答质量的影响。

当前实现中，RCTS 和 SC 使用统一的全局 embedding 配置，JE 使用独立的 Jina embedding 管线。三种策略共享同一套文档、存储、检索、LLM 和评分流程。

## 支持的切分策略

| 策略 | 名称 | 实现方式 | embedding 来源 | 说明 |
| ---- | ---- | ---- | ---- | ---- |
| RCTS | 递归字符切分 | RecursiveCharacterTextSplitter | 全局 EmbeddingService | 固定大小窗口切分 |
| SC | 语义切分 | SemanticChunker | 全局 EmbeddingService | 基于语义边界切分 |
| JE | Jina 后置切分 | Jina AI return_chunks API | Jina API | chunk 和 query 都走 Jina 模型 |

## 系统架构

```text
Client/Terminal
  -> Click CLI
  -> PostgreSQL + pgvector
  -> MinIO
  -> Embedding Provider
  -> LLM Provider
```

## 快速开始

### 环境要求

- Python 3.8+
- PostgreSQL 16+，并启用 pgvector
- MinIO，可选，用于对象存储

### 安装

1. 克隆仓库。

```bash
git clone https://github.com/TonyZhangshi81/chunk-test.git
cd chunk-test
```

1. 安装依赖。

```bash
pip install -r requirements.txt
```

当前仓库已验证可工作的 OpenAI Python SDK 版本为 [requirements.txt](requirements.txt) 中的 openai==2.36.0。

1. 创建环境变量文件。

程序会按顺序加载仓库根目录 `.env` 和 [src/.env](src/.env)，后加载的值会覆盖前者。仓库当前没有提供 `.env.example` 模板文件，请直接创建其中一个。

1. 初始化数据库。

```bash
python src/main.py rebuild-chunk-table
```

如果数据库里已经存在旧版单列 embedding_vector 结构，需要先重建 t_chunk，再重新执行 chunk。

## 配置说明

示例配置如下。

```dotenv
# 数据库
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_experiment
DB_USER=postgres
DB_PASSWORD=your_password

# MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=rag-chunks
MINIO_SECURE=False
MINIO_PATH_PATTERN=test/{uuid}/{filename}

# Embedding（用于 RCTS 和 SC）
EMBEDDING_PROVIDER=zhipu
EMBEDDING_MODEL=embedding-3
EMBEDDING_API_TYPE=zhipu
EMBEDDING_API_KEY=your_key
EMBEDDING_API_BASE=https://open.bigmodel.cn/api/paas/v4
EMBEDDING_DIMENSION=1024

# Jina（用于 JE）
JINA_API_KEY=jina_xxxxx
JINA_API_BASE=https://api.jina.ai/v1
JINA_MODEL=jina-embeddings-v2-base-zh
JINA_EMBEDDING_DIMENSION=768
JINA_POOLING_STRATEGY=mean
JINA_CHUNK_TYPE=paragraph

# LLM
LLM_API_TYPE=Qwen
LLM_MODEL=Qwen/Qwen2.5-72B-Instruct
LLM_API_KEY=your_key
LLM_API_BASE=https://api.siliconflow.cn/v1
LLM_TEMPERATURE=0.7

# Chunk
CHUNK_SIZE=500
CHUNK_OVERLAP=50
CHUNK_SC_MIN_SIZE=100
CHUNK_SC_BREAKPOINT_TYPE=percentile
CHUNK_SC_SPLIT_REGEX=(?<=[.。．?!？！、])|\n
CHUNK_JE_MIN_SIZE=100

# Search
SEARCH_TOP_K=4
```

## 向量存储说明

t_chunk 当前支持以下向量槽位。

- 2048
- 1536
- 1024
- 768
- 512
- 256
- 3072

程序会根据 embedding 实际长度写入对应列，并在检索时同时按 chunk_type、embedding_model 和向量维度做过滤。

## 使用指南

### 1. 上传文件

支持 .docx 和 .txt 格式。

```bash
python src/main.py upload --path ./docs/technical_paper.docx
```

返回值是 document_id，一行纯文本，例如：

```text
550e8400-e29b-41d4-a716-446655440000
```

### 2. 重建 Chunk 表

```bash
python src/main.py rebuild-chunk-table
```

输出示例：

```text
rebuild t_chunk completed
```

### 3. 执行切分

```bash
python src/main.py chunk --doc-id <uuid> --type RCTS
python src/main.py chunk --doc-id <uuid> --type SC
python src/main.py chunk --doc-id <uuid> --type JE
```

输出示例：

```text
chunked: 12
```

### 4. 检索问答

```bash
python src/main.py search --doc-id <uuid> --type JE --query "马斯克收购推特后做了什么"
```

输出为 JSON，例如：

```json
{
  "answer": "马斯克收购推特后，解雇了包括CEO在内的多名高管。",
  "score": 0.8428,
  "contexts": [
    "马斯克以440亿美元收购了推特。他随后解雇了包括CEO在内的多名高管。"
  ]
}
```

### 5. 对比所有策略

```bash
python src/main.py compare --doc-id <uuid> --query "什么是 RAG 系统？"
```

输出为一个按策略分组的 JSON 对象。

## JE 策略当前行为

JE 现在具备独立的 embedding 链路。

- chunk 阶段优先使用 Jina 返回的 embedding，不再回退为全局 EmbeddingService 的结果
- search 阶段使用 Jina query embedding，并只检索相同 embedding_model 的 JE 数据
- 如果 Jina 当前响应只返回 embedding、不返回 chunk 文本，系统会退化为“整文单块 + Jina embedding”模式，保证 JE 的向量维度与模型来源保持正确

最后一点是当前对 Jina 响应结构的兼容策略，不代表最终的多块切分行为已经稳定。

## 数据库设计

核心表结构如下。

```sql
t_document (
    id,
    file_name,
    file_size,
    content,
    ...
)

t_chunk (
    id,
    document_id,
    chunk_type,
    content,
    chunk_index,
    embedding_model,
    embedding_2048,
    embedding_1536,
    embedding_1024,
    embedding_768,
    embedding_512,
    embedding_256,
    embedding_3072,
    ...
)

t_experiment (
    id,
    document_id,
    chunk_type,
    query,
    answer,
    similarity_score,
    ...
)
```

## 项目结构

```text
chunk-test/
├── src/
│   ├── main.py
│   ├── config.py
│   ├── models/
│   ├── repositories/
│   ├── services/
│   │   └── chunk_strategies/
│   └── utils/
├── docker/
├── ppt/
├── requirements.txt
├── readme.md
└── Detailed-Design-Specification.md
```

## 技术栈

| 组件 | 技术 |
| ---- | ---- |
| CLI | Click |
| ORM | SQLAlchemy |
| 向量扩展 | pgvector |
| Embedding SDK | openai, zai-sdk, requests |
| 文本切分 | langchain-text-splitters |
| 对象存储 | MinIO |

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE)。
