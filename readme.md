# RAG Chunk Strategy Experiment Platform

<div align="center">

**一个用于对比不同文本切分策略对RAG系统检索质量影响的实验平台**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-336791.svg)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

## 📖 项目简介

本项目提供了一个命令行实验平台，用于在**相同文件、相同Embedding模型、相同检索流程**的前提下，对比不同文本切分（Chunk）策略对RAG系统检索命中率和回答质量的影响。

### 支持的切分策略

| 策略 | 名称 | 实现方式 | 特点 |
|------|------|----------|------|
| **RCTS** | 递归字符切分 | `RecursiveCharacterTextSplitter` | 基础方案，固定大小窗口 |
| **SC** | 语义切分 | `SemanticChunker` | 基于语义边界智能切分 |
| **JE** | Jina后置切分 | Jina AI `return_chunks` API | 直接使用 Jina 返回的语义块，独立于 RCTS/SC |

## 🏗️ 系统架构

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│  CLI (Click)│────▶│   MinIO     │
│  (Terminal) │     │             │     │  (Storage)  │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                    
                           ▼                    
                    ┌─────────────┐     ┌─────────────┐
                    │  PostgreSQL │◀────│  pgvector   │
                    │  (Database) │     │  (Vector)   │
                    └─────────────┘     └─────────────┘
                           │                    
                           ▼                    
                    ┌─────────────┐     ┌─────────────┐
                    │  LLM API    │     │Embedding API│
                    │  (智谱/通义) │     │ (OpenAI/Jina)│
                    └─────────────┘     └─────────────┘
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PostgreSQL 16+ (with pgvector)
- MinIO (可选，用于文件存储)

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/TonyZhangshi81/chunk-test.git
cd chunk-test
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

当前仓库已验证可工作的 OpenAI Python SDK 版本为 `openai==2.36.0`，详见 [requirements.txt](requirements.txt)。

3. **配置环境变量**
```bash
cp .env.example .env
# 编辑 .env 文件，填入你的API密钥和数据库配置
```

4. **初始化数据库**
```bash
python src/main.py rebuild-chunk-table
```

如果数据库里已经存在旧版 `t_chunk`（单列 `embedding_vector` 结构），需要先重建该表后再重新执行 chunk。

### 配置说明

创建 `.env` 文件并配置以下参数：

```bash
# 数据库
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_experiment
DB_USER=postgres
DB_PASSWORD=your_password

# Embedding API (用于RCTS和SC策略)
EMBEDDING_API_TYPE=openai  # openai / siliconflow
EMBEDDING_API_KEY=your_key
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_DIMENSION=1536

# Jina API (用于JE策略)
JINA_API_KEY=jina_xxxxx
JINA_API_BASE=https://api.jina.ai/v1
JINA_MODEL=jina-embeddings-v2-base
JINA_EMBEDDING_DIMENSION=768
JINA_POOLING_STRATEGY=mean
JINA_CHUNK_TYPE=paragraph

# LLM API
LLM_API_TYPE=zhipu  # openai / zhipu / qwen / deepseek
LLM_MODEL=glm-4
LLM_API_KEY=your_key

# Chunk参数
CHUNK_SIZE=500
CHUNK_OVERLAP=50
SEARCH_TOP_K=4
```

当前 `t_chunk` 支持以下向量槽位：`1536`、`1024`、`768`、`512`、`256`、`3072`。程序会根据实际返回向量长度写入对应字段，并在检索时只查询同维度、同模型的数据。

## 📋 使用指南

### 1. 上传文件

支持 `.docx` 和 `.txt` 格式：

```bash
python src/main.py upload --path ./docs/technical_paper.docx
```

### 0. 重建 Chunk 表

当你从旧版单列 `embedding_vector` 结构升级时，先执行：

```bash
python src/main.py rebuild-chunk-table
```

返回示例：
```
✓ 文件上传成功
  Document ID: 550e8400-e29b-41d4-a716-446655440000
  文件大小: 2.3 MB
```

### 2. 执行切分

对已上传的文件执行不同的切分策略：

```bash
# 递归字符切分
python src/main.py chunk --doc-id <uuid> --type RCTS

# 语义切分
python src/main.py chunk --doc-id <uuid> --type SC

# Jina 后置切分（直接使用 Jina 返回的 chunks）
python src/main.py chunk --doc-id <uuid> --type JE
```

### 3. 检索问答

```bash
python src/main.py search \
  --doc-id <uuid> \
  --type RCTS \
  --query "什么是RAG系统？"
```

输出示例：
```
📝 回答：RAG（检索增强生成）是一种结合信息检索和...
📊 质量评分：0.8523
⏱️  响应时间：1.23s
```

### 4. 对比所有策略

```bash
python src/main.py compare \
  --doc-id <uuid> \
  --query "什么是RAG系统？"
```

输出对比表格：

| 策略 | 回答质量 | 响应时间 | Chunk数量 |
|------|----------|----------|-----------|
| RCTS | 0.7821 | 1.45s | 42 |
| SC | 0.8534 | 2.31s | 28 |
| JE | 0.8912 | 3.12s | 35 |

## 🧪 实验设计

### 控制变量

- ✅ 相同的文档内容
- ✅ 相同的Embedding模型
- ✅ 相同的向量检索流程
- ✅ 相同的LLM模型和参数

### 实验变量

- 🔬 Chunk策略类型（RCTS / SC / JE）

### 评估指标

| 指标 | 计算方式 | 说明 |
|------|----------|------|
| `answer_context_similarity` | max(cos(ans, ctx_i)) | 回答与检索内容的相关性 |
| `answer_query_similarity` | cos(ans, query) | 回答与问题的相关性 |
| `combined_score` | 0.7×ctx + 0.3×query | 综合质量评分 |

## 📊 数据库设计

### 核心表结构

```sql
-- 文档表
t_document (
    id, file_name, file_size, content, ...
)

-- Chunk表（支持向量存储）
t_chunk (
    id, document_id, chunk_type, content, 
  embedding_model,
  embedding_1536(VECTOR),
  embedding_1024(VECTOR),
  embedding_768(VECTOR),
  embedding_512(VECTOR),
  embedding_256(VECTOR),
  embedding_3072(VECTOR),
  ...
)

-- 实验记录表
t_experiment (
    id, document_id, chunk_type, query, 
    answer, similarity_score, ...
)
```

## 📁 项目结构

```
chunk-test/
├── src/
│   ├── main.py                 # CLI入口
│   ├── config.py               # 配置管理
│   ├── models/                 # 数据模型
│   ├── services/               # 业务服务
│   │   ├── embedding_service.py
│   │   ├── llm_service.py
│   │   ├── storage_service.py
│   │   └── chunk_strategies/   # 三种切分策略
│   ├── repositories/           # 数据访问层
│   └── utils/                  # 工具函数
├── tests/                      # 单元测试
├── .env.example                # 配置模板
├── requirements.txt            # 依赖清单
└── README.md
```

## 🔧 技术栈

| 组件 | 技术 | 版本 |
|------|------|------|
| 运行时 | Python | 3.8+ |
| CLI框架 | Click | 8.3.0 |
| ORM | SQLAlchemy | 2.0+ |
| 数据库 | PostgreSQL | 16+ |
| 向量扩展 | pgvector | latest |
| 对象存储 | MinIO | latest |
| Embedding | OpenAI / Jina API | - |
| LLM | 智谱 / 通义 / DeepSeek | - |

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📝 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 📧 联系方式

项目维护者 - [@TonyZhangshi81](https://github.com/TonyZhangshi81)

项目链接: [https://github.com/TonyZhangshi81/chunk-test](https://github.com/TonyZhangshi81/chunk-test)

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain) - 提供文本切分工具
- [Jina AI](https://jina.ai/) - 提供Embedding API
- [pgvector](https://github.com/pgvector/pgvector) - PostgreSQL向量扩展

---

<div align="center">
  <sub>Built with ❤️ for RAG research</sub>
</div>