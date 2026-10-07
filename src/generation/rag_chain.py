"""
RAG chain orchestration.

This module connects the already-built multi-query retrieval, reranking,
context, and generation pieces into one reusable call: multi-query
retriever -> Cohere reranker -> context builder -> answer generator. It
does not load PDFs, chunk documents, enrich metadata, embed documents,
call Pinecone or OpenAI directly, or save JSON.
"""

from collections.abc import Iterator

from langchain_core.documents import Document
from langsmith import traceable

from src.context.context_builder import build_context
from src.generation.answer_generator import generate_answer, stream_answer
from src.reranking.cohere_reranker import rerank_documents
from src.retrieval.multiquery_retriever import get_multiquery_retriever
from src.utils.logger import get_logger


logger = get_logger(__name__)


@traceable(name="build_answer_sources")
def _build_sources(final_documents: list[Document]) -> list[dict]:
    """Build the source-citation list from the final, reranked documents.

    Internal format is list[dict] regardless of caller; stream_rag_query()
    converts to list-of-lists only at the UI boundary.

    Args:
        final_documents: The final, reranked documents (post rerank_documents()).

    Returns:
        One dict per document: source_number, page_number, chunk_id.
    """
    return [
        {
            "source_number": document_index + 1,
            "page_number": document.metadata.get("page_number"),
            "chunk_id": document.metadata.get("chunk_id"),
        }
        for document_index, document in enumerate(final_documents)
    ]


@traceable(name="annual_report_rag_query")
def run_rag_query(question: str) -> dict:
    """Run the full RAG flow for a question and return a structured result.

    Args:
        question: The user's question.

    Returns:
        A dict with question, answer, retrieved_chunk_count (unique
        candidate count from the MultiQueryRetriever, before reranking),
        final_context_chunk_count (count actually used to build the context
        and answer, after rerank_documents()), sources (source_number,
        page_number, chunk_id per document in the final, reranked set), and
        contexts (raw page_content per document in the final, reranked set,
        for downstream evaluation e.g. RAGAS).

    Raises:
        ValueError: If question is empty.
    """
    try:
        if not question.strip():
            raise ValueError("question must not be empty.")

        logger.info("Running RAG query for question: %s", question)

        retriever = get_multiquery_retriever()
        retrieved_documents = retriever.invoke(question)
        final_documents = rerank_documents(question, retrieved_documents)

        context = build_context(final_documents)

        answer = generate_answer(question, context)

        sources = _build_sources(final_documents)

        logger.info("Completed RAG query for question: %s", question)

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunk_count": len(retrieved_documents),
            "final_context_chunk_count": len(final_documents),
            "sources": sources,
            "contexts": [document.page_content for document in final_documents],
        }

    except Exception:
        logger.exception("Failed to run RAG query")
        raise


@traceable(name="annual_report_rag_stream_query")
def stream_rag_query(question: str) -> Iterator[tuple[str, list[list]]]:
    """Run the full RAG flow for a question, streaming the answer as it's generated.

    Retrieval and reranking run once, up front, exactly as in
    run_rag_query() - only the answer generation step streams. Source rows
    are therefore known and yielded immediately, before the first answer
    token arrives.

    Args:
        question: The user's question.

    Yields:
        (partial_answer, source_rows) tuples. The first yield has an empty
        partial_answer paired with the final source_rows, so the UI can
        display sources as soon as retrieval/reranking finishes, before the
        first answer token arrives. Every subsequent yield has
        partial_answer accumulated so far (grows with each yield) and the
        same source_rows - a list of [source_number, page_number, chunk_id]
        rows for the final, reranked document set.

    Raises:
        ValueError: If question is empty.
    """
    try:
        if not question.strip():
            raise ValueError("question must not be empty.")

        logger.info("Streaming RAG query for question: %s", question)

        retriever = get_multiquery_retriever()
        retrieved_documents = retriever.invoke(question)
        final_documents = rerank_documents(question, retrieved_documents)

        context = build_context(final_documents)

        sources = _build_sources(final_documents)
        source_rows = [
            [source["source_number"], source["page_number"], source["chunk_id"]]
            for source in sources
        ]

        partial_answer = ""

        yield partial_answer, source_rows

        for answer_chunk in stream_answer(question, context):
            partial_answer += answer_chunk
            yield partial_answer, source_rows

        logger.info("Completed streaming RAG query for question: %s", question)

    except Exception:
        logger.exception("Failed to stream RAG query")
        raise
