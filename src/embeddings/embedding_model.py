"""
Embedding model initialization utilities.

This module centralizes construction of the OpenAI embedding model used by the
embedding pipeline. It does not embed documents, create vectors, save JSON,
load PDFs, chunk documents, or call Pinecone.
"""

from langchain_openai import OpenAIEmbeddings

from src.config.settings import EMBEDDING_MODEL
from src.utils.logger import get_logger


logger = get_logger(__name__)


def get_embedding_model() -> OpenAIEmbeddings:
    """Create and return the configured OpenAI embedding model.

    Returns:
        An OpenAIEmbeddings instance configured with EMBEDDING_MODEL.

    Raises:
        ValueError: If EMBEDDING_MODEL is empty.
    """
    try:
        if not EMBEDDING_MODEL.strip():
            raise ValueError("EMBEDDING_MODEL must not be empty.")

        logger.info(
            "Initializing OpenAI embedding model %s",
            EMBEDDING_MODEL,
        )

        return OpenAIEmbeddings(model=EMBEDDING_MODEL)

    except Exception:
        logger.exception("Failed to initialize OpenAI embedding model")
        raise
