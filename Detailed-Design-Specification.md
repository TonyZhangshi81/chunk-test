
# RAG系统Chunk策略对比实验平台 - 技术设计书

## 1. 项目目标

构建 CLI 实验平台，对比三种文本切分策略对 RAG 系统检索质量的影响。

控制变量：相同文件、相同检索与评估流程、相同 LLM 配置

实验变量：Chunk 策略类型，以及策略所依赖的 embedding 管线

## 2. 技术栈

| 组件 | 技术 |
| ------ | ------ |
| 运行时 | Python 3.8+ |
| CLI框架 | Click 8.3.0 |
| ORM | SQLAlchemy 2.0+ |
| 数据库 | PostgreSQL 16+ (pgvector) |
| 对象存储 | MinIO |
| 切分工具 | langchain-text-splitters |

## 3. 三种Chunk策略

| 代码 | 名称 | 实现类 | 核心依赖 |
| ------ | ------ | -------- | --------- |
| RCTS | 递归字符切分 | `RCTSStrategy` | `RecursiveCharacterTextSplitter` |
| SC | 语义切分 | `SCStrategy` | `SemanticChunker` |
| JE | Jina后置切分 | `JEStrategy` | Jina API `return_chunks` |

## 4. 环境变量配置（.env）

程序启动时会按顺序加载仓库根目录 `.env` 和 `src/.env`，后加载的值会覆盖前者。

```bash
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

# Embedding（用于 RCTS 和 SC 策略）
EMBEDDING_PROVIDER=zhipu
EMBEDDING_MODEL=embedding-3
EMBEDDING_API_TYPE=zhipu
EMBEDDING_API_KEY=your_key
EMBEDDING_API_BASE=https://open.bigmodel.cn/api/paas/v4
EMBEDDING_DIMENSION=1024

# Jina API（用于 JE 策略）
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

# Chunk参数
CHUNK_SIZE=500
CHUNK_OVERLAP=50
CHUNK_SC_MIN_SIZE=100
CHUNK_SC_BREAKPOINT_TYPE=percentile
CHUNK_SC_SPLIT_REGEX=(?<=[.。．?!？！、])|\n
CHUNK_JE_MIN_SIZE=100

# 检索
SEARCH_TOP_K=3

# 日志
LOG_LEVEL=INFO
LOG_DIR=logs
LOG_FILE_NAME=app.log
```

日志配置说明：

- `LOG_LEVEL` 控制 CLI 输出级别，默认值为 `INFO`。
- `LOG_DIR` 指定日志目录；若为相对路径，则相对于项目根目录解析。
- `LOG_FILE_NAME` 指定日志文件名，默认值为 `app.log`。
- 日志会同时输出到控制台和文件，程序启动时自动创建缺失目录。

## 5. 数据库表结构

```sql
-- 文档表
CREATE TABLE t_document (
    id VARCHAR(50) PRIMARY KEY,
    file_name VARCHAR(1024) NOT NULL,
    file_size BIGINT,
    file_etag VARCHAR(255),
    file_type VARCHAR(100),
    mime_type VARCHAR(100),
    content TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Chunk表
CREATE TABLE t_chunk (
    id VARCHAR(50) PRIMARY KEY,
    document_id VARCHAR(50) NOT NULL REFERENCES t_document(id),
    chunk_type VARCHAR(10) NOT NULL,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    start_position INTEGER,
    end_position INTEGER,
    embedding_model VARCHAR(255),
    embedding_2048 VECTOR(2048),
    embedding_1536 VECTOR(1536),
    embedding_1024 VECTOR(1024),
    embedding_768 VECTOR(768),
    embedding_512 VECTOR(512),
    embedding_256 VECTOR(256),
    embedding_3072 VECTOR(3072),
    created_at TIMESTAMP DEFAULT NOW()
);

-- 实验记录表
CREATE TABLE t_experiment (
    id VARCHAR(50) PRIMARY KEY,
    document_id VARCHAR(50) NOT NULL REFERENCES t_document(id),
    chunk_type VARCHAR(10) NOT NULL,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    contexts TEXT,
    similarity_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## 6. 目录结构

```text
src/
├── main.py
├── config.py
├── models/
│   ├── database.py
│   ├── document.py
│   ├── chunk.py
│   └── experiment.py
├── services/
│   ├── embedding_service.py
│   ├── llm_service.py
│   ├── storage_service.py
│   └── chunk_strategies/
│       ├── __init__.py
│       ├── base.py
│       ├── rcts_strategy.py
│       ├── sc_strategy.py
│       └── je_strategy.py
├── repositories/
│   ├── document_repo.py
│   ├── chunk_repo.py
│   └── experiment_repo.py
└── utils/
    └── file_parser.py
```

## 7. CLI命令

```bash
# 上传文件
python src/main.py upload --path ./doc.docx

# 重建 Chunk 表
python src/main.py rebuild-chunk-table

# 执行切分
python src/main.py chunk --doc-id <uuid> --type RCTS
python src/main.py chunk --doc-id <uuid> --type SC
python src/main.py chunk --doc-id <uuid> --type JE

# 检索问答
python src/main.py search --doc-id <uuid> --type RCTS --query "问题内容"

# 对比所有策略
python src/main.py compare --doc-id <uuid> --query "问题内容"
```

## 8. 核心类设计

### 8.1 策略基类

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseChunkStrategy(ABC):
    
    @property
    @abstractmethod
    def strategy_type(self) -> str:
        pass
    
    @abstractmethod
    def split(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """
        返回格式:
        [
            {"content": "chunk文本", "chunk_index": 0, "start_pos": 0, "end_pos": 100},
            ...
        ]
        """
        pass
```

### 8.2 RCTS策略

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RCTSStrategy(BaseChunkStrategy):
    
    @property
    def strategy_type(self) -> str:
        return "RCTS"
    
    def split(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        chunk_size = kwargs.get('chunk_size', 500)
        overlap = kwargs.get('overlap', 50)
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
        chunks = splitter.split_text(text)
        
        return [
            {"content": chunk, "chunk_index": i}
            for i, chunk in enumerate(chunks)
        ]
```

### 8.3 SC策略

```python
from langchain.text_splitter import SemanticChunker

class SCStrategy(BaseChunkStrategy):
    
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
    
    @property
    def strategy_type(self) -> str:
        return "SC"
    
    def split(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        min_size = kwargs.get('min_chunk_size', 100)
        breakpoint_type = kwargs.get('breakpoint_type', 'percentile')
        split_regex = kwargs.get('split_regex', r'(?<=[.。．?!？！、])|\n')
        
        splitter = SemanticChunker(
            embeddings=self.embedding_service,
            breakpoint_threshold_type=breakpoint_type,
            sentence_split_regex=split_regex,
            min_chunk_size=min_size
        )
        chunks = splitter.split_text(text)
        
        return [
            {"content": chunk, "chunk_index": i}
            for i, chunk in enumerate(chunks)
        ]
```

### 8.4 JE策略

```python
import requests

class JEStrategy(BaseChunkStrategy):
    
    def __init__(self, config):
        self.api_key = config.JINA_API_KEY
        self.api_base = config.JINA_API_BASE.rstrip("/")
        self.model = config.JINA_MODEL
        self.pooling = config.JINA_POOLING_STRATEGY
        self.chunk_type = config.JINA_CHUNK_TYPE
    
    @property
    def strategy_type(self) -> str:
        return "JE"

    @property
    def embedding_model(self) -> str:
        return self.model
    
    def _get_embeddings(self, document: str) -> list[dict]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": document,
            "return_chunks": True,
            "chunk_type": self.chunk_type,
            "pooling_strategy": self.pooling
        }
        response = requests.post(
            f"{self.api_base}/embeddings",
            headers=headers,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        response_items = result.get("data", [])
        chunks = []
        for data in response_items:
            chunk_text = (data.get("text") or data.get("chunk") or "").strip()
            if not chunk_text and len(response_items) == 1:
                chunk_text = document.strip()
            if not chunk_text:
                continue
            chunks.append({
                "text": chunk_text,
                "embedding": data.get("embedding", []),
                "index": data.get("index", 0),
            })
        return chunks

    def embed_query(self, query: str) -> list[float]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": query,
            "pooling_strategy": self.pooling,
        }
        response = requests.post(
            f"{self.api_base}/embeddings",
            headers=headers,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    
    def split(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        chunks = self._get_embeddings(text)
        final_chunks = [item["text"] for item in sorted(chunks, key=lambda item: item["index"])]
        if not final_chunks:
            return [{"content": text, "chunk_index": 0}]

        payloads = [
            {"content": chunk, "chunk_index": i}
            for i, chunk in enumerate(final_chunks)
        ]
        for payload, chunk in zip(payloads, sorted(chunks, key=lambda item: item["index"])):
            payload["embedding"] = chunk["embedding"]
        return payloads
```

### 8.5 质量评估

```python
import numpy as np

class QualityEvaluator:
    """基于余弦相似度的质量评估，不调用LLM"""
    
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
    
    def evaluate(self, query: str, answer: str, contexts: List[str]) -> float:
        """
        返回0-1之间的综合质量分数
        """
        if not answer or not contexts:
            return 0.0
        
        # 获取向量
        answer_emb = self.embedding_service.embed([answer])[0]
        query_emb = self.embedding_service.embed([query])[0]
        
        # 回答与上下文的最高相似度
        context_embs = self.embedding_service.embed(contexts)
        context_sims = [
            self._cosine_similarity(answer_emb, ctx_emb)
            for ctx_emb in context_embs
        ]
        ctx_score = max(context_sims) if context_sims else 0.5
        
        # 回答与问题的相似度
        query_score = self._cosine_similarity(answer_emb, query_emb)
        
        # 综合得分
        return 0.7 * ctx_score + 0.3 * query_score
    
    def _cosine_similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

## 9. 工作流程

### 上传流程

1. 读取本地文件（支持.docx和.txt）
2. 解析为纯文本
3. 上传至MinIO（路径：`test/{uuid}/{filename}`）
4. 保存文档记录到`t_document`
5. 返回document_id

### Chunk流程

1. 根据document_id获取文档内容
2. 实例化对应的策略类
3. 调用 `split()` 获取 chunk 列表
4. 如果策略已返回 embedding（JE），优先直接使用策略返回值；否则走全局 `EmbeddingService`
5. 根据向量维度选择对应的 `embedding_xxx` 字段
6. 记录 `embedding_model`，其余未命中的向量字段保持为空
7. 批量保存到 `t_chunk`

### 检索流程

1. 为用户查询生成 embedding
2. 如果策略提供独立 `embed_query()`（JE），优先使用该查询向量；否则走全局 `EmbeddingService`
3. 根据查询向量维度选择对应的 `embedding_xxx` 字段
4. 在 `t_chunk` 中检索，过滤条件包括 `document_id`、`chunk_type`、`embedding_model`，并限定同维度字段非空
5. 获取 top_k 个最相似的 chunk
6. 构建 RAG 提示词，调用 LLM
7. 评估回答质量（余弦相似度）
8. 保存实验记录到 `t_experiment`
9. 输出回答和分数

### 数据库初始化/重建流程

1. 连接PostgreSQL后先执行`CREATE EXTENSION IF NOT EXISTS vector`
2. 新库初始化时通过SQLAlchemy `create_all()`创建`t_document`、`t_chunk`、`t_experiment`
3. 如果数据库里仍是旧版单列`embedding_vector`结构，执行`python src/main.py rebuild-chunk-table`重建`t_chunk`
4. 重建后需要重新执行chunk，旧的chunk数据不做自动迁移

## 10. requirements.txt

```text
click==8.3.0
sqlalchemy==2.0.25
psycopg2-binary==2.9.11
pgvector==0.2.5
minio==7.2.5
python-docx==1.1.0
openai==2.36.0
zai-sdk==0.2.2
requests==2.31.0
langchain-text-splitters==0.3.11
numpy==2.4.4
python-dotenv==1.0.1
```
