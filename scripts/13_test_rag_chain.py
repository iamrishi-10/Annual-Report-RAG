"""
Test the reusable run_rag_query() chain over sample questions.

This script does not wire the retriever, context builder, or answer
generator manually — that wiring lives in run_rag_query() itself.
"""

from dotenv import load_dotenv

from src.config.settings import PROCESSED_DATA_DIR, PROJECT_ROOT
from src.generation.rag_chain import run_rag_query
from src.utils.json_writer import save_json
from src.utils.logger import get_logger


logger = get_logger(__name__)

SAMPLE_QUESTIONS = [
    "What was Infosys revenue in 2025?",
    "What are the key AI initiatives Infosys discusses in the annual report",
    "Who is the CEO of Infosys?",
    "What are the key business segments?",
]


def main() -> None:
    """Run run_rag_query() over sample questions and save the results."""
    load_dotenv(PROJECT_ROOT / ".env")

    rag_chain_preview = []

    for question in SAMPLE_QUESTIONS:
        logger.info("Running RAG chain for question: %s", question)

        result = run_rag_query(question)

        rag_chain_preview.append(result)

    output_path = PROCESSED_DATA_DIR / "rag_chain_preview.json"

    save_json(rag_chain_preview, output_path)

    logger.info(
        "Saved RAG chain preview for %d question(s) to %s",
        len(rag_chain_preview),
        output_path,
    )


if __name__ == "__main__":
    main()
