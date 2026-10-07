"""
Test whether Pinecone retrieval returns relevant chunks for sample questions.

This script does not call the LLM. It only exercises the retriever and saves
what came back for manual inspection.
"""

from dotenv import load_dotenv

from src.config.settings import PROCESSED_DATA_DIR, PROJECT_ROOT
from src.retrieval.retriever import get_retriever
from src.utils.json_writer import save_json
from src.utils.logger import get_logger


logger = get_logger(__name__)

SAMPLE_QUESTIONS = [
    "What was Infosys revenue in 2025?",
    "What is the company's approach to AI?",
    "Who is the CEO of Infosys?",
    "What are the key business segments?",
]


def main() -> None:
    """Retrieve chunks for sample questions and save a retrieval preview."""
    load_dotenv(PROJECT_ROOT / ".env")

    retriever = get_retriever()

    retrieval_preview = []

    for question in SAMPLE_QUESTIONS:
        logger.info("Retrieving chunks for question: %s", question)

        retrieved_documents = retriever.invoke(question)

        retrieval_preview.append(
            {
                "question": question,
                "retrieved_chunk_count": len(retrieved_documents),
                "retrieved_chunks": [
                    {
                        "page_content": document.page_content,
                        "metadata": document.metadata,
                    }
                    for document in retrieved_documents
                ],
            }
        )

    output_path = PROCESSED_DATA_DIR / "retrieval_preview.json"

    save_json(retrieval_preview, output_path)

    logger.info(
        "Saved retrieval preview for %d question(s) to %s",
        len(retrieval_preview),
        output_path,
    )


if __name__ == "__main__":
    main()
