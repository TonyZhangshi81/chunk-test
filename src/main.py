import json
import mimetypes
from pathlib import Path
from uuid import uuid4

import click

from config import config
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


def _build_runtime():
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
    runtime["session"].close()


def _require_document(document_repo: DocumentRepository, document_id: str) -> Document:
    document = document_repo.get_by_id(document_id)
    if document is None:
        raise click.ClickException(f"Document not found: {document_id}")
    return document


def _chunk_kwargs(chunk_type: str) -> dict:
    return {
        "chunk_size": config.CHUNK_SIZE,
        "overlap": config.CHUNK_OVERLAP,
        "min_chunk_size": config.CHUNK_SC_MIN_SIZE if chunk_type == "SC" else config.CHUNK_JE_MIN_SIZE,
        "breakpoint_type": config.CHUNK_SC_BREAKPOINT_TYPE,
        "split_regex": config.CHUNK_SC_SPLIT_REGEX,
    }


def _resolve_chunk_embeddings(runtime: dict, strategy, chunk_payloads: list[dict]) -> tuple[list[list[float]], str]:
    if chunk_payloads and all(item.get("embedding") is not None for item in chunk_payloads):
        embeddings = [item["embedding"] for item in chunk_payloads]
        embedding_model = getattr(strategy, "embedding_model", config.EMBEDDING_MODEL)
        return embeddings, embedding_model

    embeddings = runtime["embedding_service"].embed([item["content"] for item in chunk_payloads])
    return embeddings, config.EMBEDDING_MODEL


def _resolve_query_embedding(runtime: dict, strategy, query: str) -> tuple[list[float], str]:
    if hasattr(strategy, "embed_query"):
        return strategy.embed_query(query), getattr(strategy, "embedding_model", config.EMBEDDING_MODEL)
    return runtime["embedding_service"].embed_query(query), config.EMBEDDING_MODEL


def _chunk_document(runtime: dict, document_id: str, chunk_type: str) -> int:
    document = _require_document(runtime["document_repo"], document_id)
    if not document.content:
        raise click.ClickException("Document content is empty")

    strategy = build_strategy(chunk_type, config, runtime["embedding_service"])
    chunk_payloads = strategy.split(document.content, **_chunk_kwargs(chunk_type))
    embeddings, embedding_model = _resolve_chunk_embeddings(runtime, strategy, chunk_payloads)
    if embeddings:
        get_embedding_column_name(len(embeddings[0]))

    chunk_models = []
    for item, embedding in zip(chunk_payloads, embeddings):
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
    return len(chunk_models)


def _search_document(runtime: dict, document_id: str, chunk_type: str, query: str) -> dict:
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
        raise click.ClickException(
            (
                f"No chunks found for document_id={document_id}, chunk_type={chunk_type}, "
                f"embedding_model={embedding_model}. Run chunk first."
            )
        )

    contexts = [chunk.content for chunk in chunks]
    answer = runtime["llm_service"].answer(query, contexts)
    score = runtime["quality_evaluator"].evaluate(query, answer, contexts)

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
    return {"answer": answer, "score": score, "contexts": contexts}


@click.group()
def cli() -> None:
    pass


@cli.command("rebuild-chunk-table")
def rebuild_chunk_table_command() -> None:
    rebuild_chunk_table()
    click.echo("rebuild t_chunk completed")


@cli.command()
@click.option("--path", "file_path", required=True, type=click.Path(exists=True, dir_okay=False))
def upload(file_path: str) -> None:
    runtime = _build_runtime()
    runtime["quality_evaluator"] = QualityEvaluator(runtime["embedding_service"])
    try:
        path = Path(file_path)
        content, raw_bytes = parse_file(file_path)
        document_id = str(uuid4())
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
        click.echo(document_id)
    finally:
        _close_runtime(runtime)


@cli.command()
@click.option("--doc-id", required=True)
@click.option("--type", "chunk_type", required=True, type=click.Choice(["RCTS", "SC", "JE"], case_sensitive=False))
def chunk(doc_id: str, chunk_type: str) -> None:
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
    cli()