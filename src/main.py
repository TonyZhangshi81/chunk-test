import json
import logging
import mimetypes
from pathlib import Path
from uuid import uuid4

import click

from config import config
from logging_config import setup_logging
from models.chunk import Chunk, build_embedding_column_values, get_embedding_column_name
from models.database import get_session, init_db, rebuild_chunk_table
from models.document import Document
from models.experiment import Experiment
from repositories.chunk_repo import ChunkRepository
from repositories.document_repo import DocumentRepository
from repositories.experiment_repo import ExperimentRepository
from services.chunk_strategies import build_strategy
from services.embedding_service import EmbeddingService
from services.llm_service import LLMService
from services.quality_evaluator import QualityEvaluator
from services.storage_service import StorageService
from utils.file_parser import parse_file


logger = logging.getLogger(__name__)


def _build_runtime():
    """为一次 CLI 调用创建共享的运行时依赖。"""
    logger.info("Initializing runtime dependencies")
    # 按需初始化表结构，确保每个 CLI 命令都能独立启动运行。
    init_db()
    session = get_session()
    return {
        "session": session,
        "document_repo": DocumentRepository(session),
        "chunk_repo": ChunkRepository(session),
        "experiment_repo": ExperimentRepository(session),
        "embedding_service": EmbeddingService(config),
        "llm_service": LLMService(config),
        "storage_service": StorageService(config),
    }


def _close_runtime(runtime: dict) -> None:
    """在 CLI 命令结束后释放数据库会话。"""
    logger.info("Closing runtime session")
    runtime["session"].close()


def _require_document(document_repo: DocumentRepository, document_id: str) -> Document:
    """加载文档，不存在时抛出适合 CLI 展示的错误。"""
    document = document_repo.get_by_id(document_id)
    if document is None:
        logger.warning("Document lookup failed for document_id=%s", document_id)
        raise click.ClickException(f"Document not found: {document_id}")
    return document


def _chunk_kwargs(chunk_type: str) -> dict:
    """根据当前配置组装切块策略参数。"""
    return {
        # 只有 SC 会直接消费这些本地语义切分参数；JE 主要依赖远端 Jina 配置。
        "chunk_size": config.CHUNK_SIZE,
        "overlap": config.CHUNK_OVERLAP,
        "min_chunk_size": config.CHUNK_SC_MIN_SIZE if chunk_type == "SC" else config.CHUNK_JE_MIN_SIZE,
        "breakpoint_type": config.CHUNK_SC_BREAKPOINT_TYPE,
        "split_regex": config.CHUNK_SC_SPLIT_REGEX,
    }


def _resolve_chunk_embeddings(runtime: dict, strategy, chunk_payloads: list[dict]) -> tuple[list[list[float]], str]:
    """优先复用策略自带向量，否则退回共享 embedding 服务。"""
    if chunk_payloads and all(item.get("embedding") is not None for item in chunk_payloads):
        # JE 在 split 阶段就返回向量，若此处再次 embedding 会额外花费成本并混用模型。
        embeddings = [item["embedding"] for item in chunk_payloads]
        embedding_model = getattr(strategy, "embedding_model", config.EMBEDDING_MODEL)
        logger.info(
            "Using strategy-provided embeddings for %s chunks with model=%s",
            len(embeddings),
            embedding_model,
        )
        return embeddings, embedding_model

    logger.info("Generating %s chunk embeddings via shared embedding service", len(chunk_payloads))
    embeddings = runtime["embedding_service"].embed([item["content"] for item in chunk_payloads])
    return embeddings, config.EMBEDDING_MODEL


def _resolve_query_embedding(runtime: dict, strategy, query: str) -> tuple[list[float], str]:
    """解析检索阶段应使用的查询向量。"""
    if hasattr(strategy, "embed_query"):
        embedding_model = getattr(strategy, "embedding_model", config.EMBEDDING_MODEL)
        logger.info("Using strategy-specific query embedding with model=%s", embedding_model)
        return strategy.embed_query(query), embedding_model
    logger.info("Using shared embedding service for query embedding")
    return runtime["embedding_service"].embed_query(query), config.EMBEDDING_MODEL


def _chunk_document(runtime: dict, document_id: str, chunk_type: str) -> int:
    """执行文档切块、向量生成，并持久化结果。"""
    logger.info("Starting chunk flow for document_id=%s chunk_type=%s", document_id, chunk_type)
    document = _require_document(runtime["document_repo"], document_id)
    if not document.content:
        logger.warning("Document content is empty for document_id=%s", document_id)
        raise click.ClickException("Document content is empty")

    strategy = build_strategy(chunk_type, config, runtime["embedding_service"])
    chunk_payloads = strategy.split(document.content, **_chunk_kwargs(chunk_type))
    embeddings, embedding_model = _resolve_chunk_embeddings(runtime, strategy, chunk_payloads)
    if embeddings:
        # 这里先校验向量维度，避免把数据写进数据库不支持的向量列。
        get_embedding_column_name(len(embeddings[0]))
        logger.info(
            "Validated embedding dimension=%s for document_id=%s chunk_type=%s",
            len(embeddings[0]),
            document_id,
            chunk_type,
        )

    chunk_models = []
    for item, embedding in zip(chunk_payloads, embeddings):
        # 在进入仓储层前把 chunk 元数据和向量列值一次性组装完整。
        chunk_models.append(
            Chunk(
                id=str(uuid4()),
                document_id=document_id,
                chunk_type=chunk_type,
                content=item["content"],
                chunk_index=item["chunk_index"],
                start_position=item.get("start_pos"),
                end_position=item.get("end_pos"),
                embedding_model=embedding_model,
                **build_embedding_column_values(embedding),
            )
        )
    runtime["chunk_repo"].replace_for_document(document_id, chunk_type, chunk_models)
    logger.info(
        "Finished chunk flow for document_id=%s chunk_type=%s chunk_count=%s",
        document_id,
        chunk_type,
        len(chunk_models),
    )
    return len(chunk_models)


def _search_document(runtime: dict, document_id: str, chunk_type: str, query: str) -> dict:
    """执行单一策略下的检索、回答生成与质量评分。"""
    logger.info("Starting search flow for document_id=%s chunk_type=%s", document_id, chunk_type)
    _require_document(runtime["document_repo"], document_id)
    strategy = build_strategy(chunk_type, config, runtime["embedding_service"])
    query_embedding, embedding_model = _resolve_query_embedding(runtime, strategy, query)
    chunks = runtime["chunk_repo"].search_similar(
        document_id,
        chunk_type,
        embedding_model,
        query_embedding,
        config.SEARCH_TOP_K,
    )
    if not chunks:
        logger.warning(
            "No chunks found for document_id=%s chunk_type=%s model=%s",
            document_id,
            chunk_type,
            embedding_model,
        )
        raise click.ClickException(
            (
                f"No chunks found for document_id={document_id}, chunk_type={chunk_type}, "
                f"embedding_model={embedding_model}. Run chunk first."
            )
        )

    # 检索得到的 chunk 文本是回答阶段唯一可用的上下文输入。
    contexts = [chunk.content for chunk in chunks]
    answer = runtime["llm_service"].answer(query, contexts)
    score = runtime["quality_evaluator"].evaluate(query, answer, contexts)

    # 每次检索都会落成一条实验记录，方便后续做跨策略对比。
    experiment = Experiment(
        id=str(uuid4()),
        document_id=document_id,
        chunk_type=chunk_type,
        query=query,
        answer=answer,
        contexts=json.dumps(contexts, ensure_ascii=False),
        similarity_score=score,
    )
    runtime["experiment_repo"].create(experiment)
    logger.info(
        "Finished search flow for document_id=%s chunk_type=%s contexts=%s score=%.4f",
        document_id,
        chunk_type,
        len(contexts),
        score,
    )
    return {"answer": answer, "score": score, "contexts": contexts}


@click.group()
def cli() -> None:
    """切块策略实验平台的 CLI 入口。"""


@cli.command("rebuild-chunk-table")
def rebuild_chunk_table_command() -> None:
    """重建 chunk 表及其相关向量列。"""
    logger.info("Rebuilding chunk table")
    rebuild_chunk_table()
    click.echo("rebuild t_chunk completed")


@cli.command()
@click.option("--path", "file_path", required=True, type=click.Path(exists=True, dir_okay=False))
def upload(file_path: str) -> None:
    """上传文档文件，保存元数据，并将原始文件写入 MinIO。"""
    runtime = _build_runtime()
    runtime["quality_evaluator"] = QualityEvaluator(runtime["embedding_service"])
    try:
        logger.info("Starting upload flow for path=%s", file_path)
        path = Path(file_path)
        content, raw_bytes = parse_file(file_path)
        document_id = str(uuid4())
        # 对象存储和关系表共用同一个 document_id，便于后续追踪同一份文档。
        object_name = config.MINIO_PATH_PATTERN.format(uuid=document_id, filename=path.name)
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        file_etag = runtime["storage_service"].upload_bytes(object_name, raw_bytes, mime_type)

        document = Document(
            id=document_id,
            file_name=path.name,
            file_size=path.stat().st_size,
            file_etag=file_etag,
            file_type=path.suffix.lstrip("."),
            mime_type=mime_type,
            content=content,
        )
        runtime["document_repo"].create(document)
        logger.info("Upload flow completed for file_name=%s document_id=%s", path.name, document_id)
        click.echo(document_id)
    finally:
        _close_runtime(runtime)


@cli.command()
@click.option("--doc-id", required=True)
@click.option("--type", "chunk_type", required=True, type=click.Choice(["RCTS", "SC", "JE"], case_sensitive=False))
def chunk(doc_id: str, chunk_type: str) -> None:
    """使用指定策略为文档生成 chunk。"""
    runtime = _build_runtime()
    runtime["quality_evaluator"] = QualityEvaluator(runtime["embedding_service"])
    try:
        count = _chunk_document(runtime, doc_id, chunk_type.upper())
        click.echo(f"chunked: {count}")
    finally:
        _close_runtime(runtime)


@cli.command()
@click.option("--doc-id", required=True)
@click.option("--type", "chunk_type", required=True, type=click.Choice(["RCTS", "SC", "JE"], case_sensitive=False))
@click.option("--query", required=True)
def search(doc_id: str, chunk_type: str, query: str) -> None:
    """在已保存 chunk 中执行检索，并生成问题答案。"""
    runtime = _build_runtime()
    runtime["quality_evaluator"] = QualityEvaluator(runtime["embedding_service"])
    try:
        result = _search_document(runtime, doc_id, chunk_type.upper(), query)
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        _close_runtime(runtime)


@cli.command()
@click.option("--doc-id", required=True)
@click.option("--query", required=True)
def compare(doc_id: str, query: str) -> None:
    """用同一个问题依次运行全部切块策略。"""
    runtime = _build_runtime()
    runtime["quality_evaluator"] = QualityEvaluator(runtime["embedding_service"])
    try:
        results = {}
        for chunk_type in ["RCTS", "SC", "JE"]:
            results[chunk_type] = _search_document(runtime, doc_id, chunk_type, query)
        click.echo(json.dumps(results, ensure_ascii=False, indent=2))
    finally:
        _close_runtime(runtime)


if __name__ == "__main__":
    setup_logging(config.LOG_LEVEL, config.log_file_path)
    cli()