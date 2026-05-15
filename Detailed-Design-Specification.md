
# RAG系统Chunk策略对比实验平台 - 技术设计书

## 1. 项目目标

构建CLI实验平台，对比三种文本切分策略对RAG系统检索质量的影响。

**控制变量**：相同文件、相同Embedding模型、相同检索流程
**实验变量**：Chunk策略类型

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
| JE | Jina语义切分 | `JEStrategy` | Jina API (`JinaEmbeddings`) |

## 4. 环境变量配置（.env）

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

# Embedding（用于RCTS和SC策略）
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_API_TYPE=openai
EMBEDDING_API_KEY=your_key
EMBEDDING_API_BASE=https://api.openai.com/v1
EMBEDDING_DIMENSION=1536

# Jina API（用于JE策略）
JINA_API_KEY=jina_xxxxx
JINA_API_BASE=https://api.jina.ai/v1
JINA_MODEL=jina-embeddings-v2-base
JINA_EMBEDDING_DIMENSION=768
JINA_POOLING_STRATEGY=mean

# LLM
LLM_API_TYPE=zhipu
LLM_MODEL=glm-4
LLM_API_KEY=your_key
LLM_API_BASE=https://open.bigmodel.cn/api/paas/v4
LLM_TEMPERATURE=0.7

# Chunk参数
CHUNK_SIZE=500
CHUNK_OVERLAP=50
CHUNK_SC_MIN_SIZE=100
CHUNK_SC_BREAKPOINT_TYPE=percentile
CHUNK_SC_SPLIT_REGEX=(?<=[.。．?!？！、])|\n
CHUNK_JE_MIN_SIZE=100

# 检索
SEARCH_TOP_K=4
```

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
from langchain.text_splitter import RecursiveCharacterTextSplitter

class JEStrategy(BaseChunkStrategy):
    
    def __init__(self, config):
        self.api_key = config.JINA_API_KEY
        self.api_base = config.JINA_API_BASE
        self.model = config.JINA_MODEL
        self.pooling = config.JINA_POOLING_STRATEGY
    
    @property
    def strategy_type(self) -> str:
        return "JE"
    
    def _get_embeddings(self, texts: list) -> list:
        """调用Jina API获取向量"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": texts,
            "pooling_strategy": self.pooling
        }
        response = requests.post(
            f"{self.api_base}/embeddings",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]
    
    def _cosine_similarity(self, a, b):
        import numpy as np
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _split_sentences(self, text: str, regex: str) -> list:
        import re
        sentences = re.split(regex, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def split(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        chunk_size = kwargs.get('chunk_size', 500)
        overlap = kwargs.get('overlap', 50)
        min_size = kwargs.get('min_chunk_size', 100)
        split_regex = kwargs.get('split_regex', r'(?<=[.。．?!？！、])|\n')
        
        # 1. 句子分割
        sentences = self._split_sentences(text, split_regex)
        
        if len(sentences) <= 1:
            return [{"content": text, "chunk_index": 0}]
        
        # 2. 获取句子向量
        embeddings = self._get_embeddings(sentences)
        
        # 3. 计算相邻句子相似度，找断点
        similarities = []
        for i in range(len(embeddings) - 1):
            similarities.append(self._cosine_similarity(embeddings[i], embeddings[i+1]))
        
        # 4. 相似度低于均值减半标准差的位置作为断点
        import numpy as np
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        threshold = mean_sim - 0.5 * std_sim
        
        breakpoints = []
        current_size = 0
        for i, sim in enumerate(similarities):
            current_size += len(sentences[i])
            if sim < threshold and current_size >= min_size:
                breakpoints.append(i + 1)
                current_size = 0
        
        # 5. 按断点切分
        chunks = []
        start = 0
        for end in sorted(set(breakpoints + [len(sentences)])):
            if end > start:
                chunk_text = "。".join(sentences[start:end])
                chunks.append(chunk_text)
                start = end
        
        # 6. 处理过大的chunk
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_size:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=overlap
                )
                final_chunks.extend(splitter.split_text(chunk))
            else:
                final_chunks.append(chunk)
        
        return [
            {"content": chunk, "chunk_index": i}
            for i, chunk in enumerate(final_chunks)
        ]
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
3. 调用`split()`获取chunk列表
4. 为每个chunk生成embedding，并根据向量维度选择对应的`embedding_xxx`字段
5. 记录`embedding_model`，其余未命中的向量字段保持为空
6. 批量保存到`t_chunk`

### 检索流程

1. 对用户查询生成embedding
2. 根据查询向量维度选择对应的`embedding_xxx`字段
3. 在`t_chunk`中检索（按document_id、chunk_type、embedding_model过滤，并限定同维度字段非空）
4. 获取top_k个最相似的chunk
5. 构建RAG提示词，调用LLM
6. 评估回答质量（余弦相似度）
7. 保存实验记录到`t_experiment`
8. 输出回答和分数

### 数据库初始化/重建流程

1. 连接PostgreSQL后先执行`CREATE EXTENSION IF NOT EXISTS vector`
2. 新库初始化时通过SQLAlchemy `create_all()`创建`t_document`、`t_chunk`、`t_experiment`
3. 如果数据库里仍是旧版单列`embedding_vector`结构，执行`python src/main.py rebuild-chunk-table`重建`t_chunk`
4. 重建后需要重新执行chunk，旧的chunk数据不做自动迁移

## 10. requirements.txt

```text
click==8.3.0
sqlalchemy==2.0.25
psycopg2-binary==2.9.9
pgvector==0.2.5
minio==7.2.5
python-docx==1.1.0
openai==2.36.0
requests==2.31.0
langchain-text-splitters==0.0.1
numpy==1.24.3
python-dotenv==1.0.1
```
