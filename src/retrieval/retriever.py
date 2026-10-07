"""
Dense candidate retrieval utilities.

This module builds a LangChain retriever backed by the existing Pinecone
index and embedding model. It does not load PDFs, chunk documents, clean
metadata, index documents, call the LLM, or format final answers.
"""

from langchain_core.vectorstores import VectorStoreRetriever

from src.config.settings import SEARCH_TYPE, TOP_K
from src.utils.logger import get_logger
from src.vectorstore.vectorstore_indexer import get_vector_store


logger = get_logger(__name__)


def get_retriever() -> VectorStoreRetriever:
    """Build a retriever over the existing Pinecone index.

    Returns:
        A VectorStoreRetriever configured with SEARCH_TYPE and TOP_K from
        settings, scoped to PINECONE_NAMESPACE.
    """
    try:
        logger.info("Building retriever with search_type=%s, top_k=%d", SEARCH_TYPE, TOP_K)

        vector_store = get_vector_store()

        retriever = vector_store.as_retriever(
            search_type=SEARCH_TYPE,
            search_kwargs={"k": TOP_K},
        )

        logger.info("Retriever ready")

        return retriever

    except Exception:
        logger.exception("Failed to build retriever")
        raise
