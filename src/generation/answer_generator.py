"""
Answer generation utilities.

This module builds the RAG prompt from a question and already-formatted
context, calls the LLM, and returns the final answer string. It does not
load PDFs, chunk documents, embed or index documents, call Pinecone
directly, or perform retrieval — context is always supplied by the caller.
"""

from collections.abc import Iterator

from src.config.prompts import get_rag_prompt
from src.generation.llm_client import get_llm
from src.utils.logger import get_logger


logger = get_logger(__name__)


def generate_answer(question: str, context: str) -> str:
    """Generate an answer to a question using already-retrieved context.

    Args:
        question: The user's question.
        context: Formatted context string (e.g. from build_context()).

    Returns:
        The LLM's answer text.

    Raises:
        ValueError: If question or context is empty.
    """
    try:
        if not question.strip():
            raise ValueError("question must not be empty.")

        if not context.strip():
            raise ValueError("context must not be empty.")

        logger.info("Generating answer for question: %s", question)

        prompt = get_rag_prompt()
        llm = get_llm()
        chain = prompt | llm

        response = chain.invoke({"context": context, "question": question})

        logger.info("Generated answer for question: %s", question)

        return response.content

    except Exception:
        logger.exception("Failed to generate answer")
        raise


def stream_answer(question: str, context: str) -> Iterator[str]:
    """Stream an answer to a question using already-retrieved context.

    Same prompt and LLM as generate_answer(), but yields answer text
    incrementally as the LLM produces it instead of waiting for the full
    response.

    Args:
        question: The user's question.
        context: Formatted context string (e.g. from build_context()).

    Yields:
        Successive chunks of the LLM's answer text, in order. Concatenating
        all yielded chunks reproduces the same answer generate_answer()
        would return.

    Raises:
        ValueError: If question or context is empty.
    """
    try:
        if not question.strip():
            raise ValueError("question must not be empty.")

        if not context.strip():
            raise ValueError("context must not be empty.")

        logger.info("Streaming answer for question: %s", question)

        prompt = get_rag_prompt()
        llm = get_llm()
        chain = prompt | llm

        for chunk in chain.stream({"context": context, "question": question}):
            if chunk.content:
                yield chunk.content

        logger.info("Finished streaming answer for question: %s", question)

    except Exception:
        logger.exception("Failed to stream answer")
        raise
