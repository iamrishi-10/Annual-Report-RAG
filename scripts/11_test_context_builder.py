"""
Test whether retrieved chunks format into a clean context string.

This script does not call the LLM. It only exercises the retriever and
build_context() and saves the formatted context for manual inspection.
"""

from dotenv import load_dotenv

from src.config.settings import PROCESSED_DATA_DIR, PROJECT_ROOT
from src.context.context_builder import build_context
from src.retrieval.retriever import get_retriever
from src.utils.logger import get_logger


logger = get_logger(__name__)

SAMPLE_QUESTION = "What was Infosys revenue in 2025?"


def main() -> None:
    """Retrieve chunks for one sample question and save a context preview."""
    load_dotenv(PROJECT_ROOT / ".env")

    retriever = get_retriever()

    logger.info("Retrieving chunks for question: %s", SAMPLE_QUESTION)

    retrieved_documents = retriever.invoke(SAMPLE_QUESTION)

    context = build_context(retrieved_documents)

    output_path = PROCESSED_DATA_DIR / "context_preview.txt"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(context, encoding="utf-8")

    logger.info(
        "Saved context preview for %d retrieved chunk(s) to %s",
        len(retrieved_documents),
        output_path,
    )


if __name__ == "__main__":
    main()
