"""
Multi-query retrieval utilities.

This module wraps the existing Pinecone retriever with LangChain's
built-in MultiQueryRetriever, which uses the LLM to generate multiple
phrasings of a question and returns the unique union of documents retrieved
across all of them. It does not rerank, build context, generate the final
answer, save JSON, or call Pinecone directly.
"""

from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from src.generation.llm_client import get_llm
from src.retrieval.retriever import get_retriever
from src.utils.logger import get_logger


logger = get_logger(__name__)


def get_multiquery_retriever() -> MultiQueryRetriever:
    """Build a MultiQueryRetriever wrapping the existing Pinecone retriever.

    Returns:
        A MultiQueryRetriever that generates multiple query phrasings via
        the LLM, retrieves candidates for each with the base retriever, and
        returns the unique union (including the original query).
    """
    try:
        logger.info("Building multi-query retriever")

        base_retriever = get_retriever()
        llm = get_llm()

        multiquery_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm,
            include_original=True,
        )

        logger.info("Multi-query retriever ready")

        return multiquery_retriever

    except Exception:
        logger.exception("Failed to build multi-query retriever")
        raise
