"""
Cohere reranking utilities.

This module reranks candidate Documents a retriever already returned,
using Cohere's hosted rerank model, and keeps the top FINAL_TOP_K. It does
not call Pinecone, perform retrieval, build context, or generate answers.
"""

from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from langsmith import traceable

from src.config.api_keys import is_key_set
from src.config.settings import FINAL_TOP_K, RERANKING_MODEL
from src.utils.logger import get_logger


logger = get_logger(__name__)


@traceable(name="cohere_rerank_documents")
def rerank_documents(question: str, documents: list[Document]) -> list[Document]:
    """Rerank candidate documents against the question and keep the top FINAL_TOP_K.

    Args:
        question: The user's question.
        documents: Candidate Documents returned by the retriever.

    Returns:
        Up to FINAL_TOP_K Documents, ordered by Cohere relevance score
        descending. Original metadata is preserved (plus a relevance_score
        key Cohere adds).

    Raises:
        ValueError: If question or documents is empty, FINAL_TOP_K is not
            greater than 0, or RERANKING_MODEL is empty.
    """
    try:
        if not question.strip():
            raise ValueError("question must not be empty.")

        if not documents:
            raise ValueError("documents must not be empty.")

        if FINAL_TOP_K <= 0:
            raise ValueError("FINAL_TOP_K must be greater than 0.")

        if not RERANKING_MODEL.strip():
            raise ValueError("RERANKING_MODEL must not be empty.")

        if not is_key_set("COHERE_API_KEY"):
            raise ValueError("COHERE_API_KEY must be set.")

        logger.info(
            "Reranking %d candidate document(s) for question: %s",
            len(documents),
            question,
        )

        reranker = CohereRerank(model=RERANKING_MODEL, top_n=FINAL_TOP_K)

        reranked_documents = list(reranker.compress_documents(documents, question))

        logger.info(
            "Reranked to %d final document(s) for question: %s",
            len(reranked_documents),
            question,
        )

        return reranked_documents

    except Exception:
        logger.exception("Failed to rerank documents")
        raise
