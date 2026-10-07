"""
Chat model client utilities.

This module centralizes construction of the OpenAI chat model used for
answer generation. It does not build prompts, perform retrieval, build
context, format final answers, or call Pinecone.
"""

from langchain_openai import ChatOpenAI

from src.config.settings import GENERATION_MODEL
from src.utils.logger import get_logger


logger = get_logger(__name__)


def get_llm() -> ChatOpenAI:
    """Create and return the configured OpenAI chat model.

    Returns:
        A ChatOpenAI instance configured with GENERATION_MODEL.

    Raises:
        ValueError: If GENERATION_MODEL is empty.
    """
    try:
        if not GENERATION_MODEL.strip():
            raise ValueError("GENERATION_MODEL must not be empty.")

        logger.info("Initializing OpenAI chat model %s", GENERATION_MODEL)

        return ChatOpenAI(model=GENERATION_MODEL)

    except Exception:
        logger.exception("Failed to initialize OpenAI chat model")
        raise
