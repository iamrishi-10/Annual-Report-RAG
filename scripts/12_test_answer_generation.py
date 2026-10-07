"""
Test the full RAG flow: question -> retrieval -> context -> answer generation.

This is the first script that calls the LLM.
"""

from dotenv import load_dotenv

from src.config.settings import PROCESSED_DATA_DIR, PROJECT_ROOT
from src.context.context_builder import build_context
from src.generation.answer_generator import generate_answer
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
    """Run the full retrieval -> context -> generation flow and save a preview."""
    load_dotenv(PROJECT_ROOT / ".env")

    retriever = get_retriever()

    answer_generation_preview = []

    for question in SAMPLE_QUESTIONS:
        logger.info("Running full RAG flow for question: %s", question)

        retrieved_documents = retriever.invoke(question)
        context = build_context(retrieved_documents)
        answer = generate_answer(question, context)

        answer_generation_preview.append(
            {
                "question": question,
                "retrieved_chunk_count": len(retrieved_documents),
                "context": context,
                "answer": answer,
            }
        )

    output_path = PROCESSED_DATA_DIR / "answer_generation_preview.json"

    save_json(answer_generation_preview, output_path)

    logger.info(
        "Saved answer generation preview for %d question(s) to %s",
        len(answer_generation_preview),
        output_path,
    )


if __name__ == "__main__":
    main()
