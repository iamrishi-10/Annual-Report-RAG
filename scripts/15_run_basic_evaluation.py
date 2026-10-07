"""
Run golden questions through the RAG chain and score them with basic_evaluator.

For now this only runs the first MAX_QUESTIONS golden questions, not the
full 100-question dataset.
"""

from dotenv import load_dotenv

from src.config.settings import EVALUATION_DATA_DIR, PROJECT_ROOT
from src.evaluation.basic_evaluator import (
    evaluate_single_result,
    load_golden_questions,
    summarize_evaluation_results,
)
from src.generation.rag_chain import run_rag_query
from src.utils.json_writer import save_json
from src.utils.logger import get_logger


logger = get_logger(__name__)

MAX_QUESTIONS = 30


def main() -> None:
    """Run MAX_QUESTIONS golden questions through the RAG chain and score them."""
    load_dotenv(PROJECT_ROOT / ".env")

    golden_questions = load_golden_questions()[:MAX_QUESTIONS]

    results = []

    for golden_item in golden_questions:
        logger.info(
            "Evaluating question id=%s: %s", golden_item["id"], golden_item["question"]
        )

        try:
            rag_result = run_rag_query(golden_item["question"])
            result = evaluate_single_result(golden_item, rag_result)

        except Exception as error:
            logger.exception(
                "Failed to evaluate question id=%s", golden_item["id"]
            )
            result = {
                "id": golden_item["id"],
                "question": golden_item["question"],
                "category": golden_item["category"],
                "difficulty": golden_item["difficulty"],
                "error": str(error),
                "answered": False,
                "source_page_hit": False,
                "contexts": [],
                "retrieved_chunk_count": 0,
                "final_context_chunk_count": 0,
            }

        results.append(result)

    summary = summarize_evaluation_results(results)

    results_output_path = EVALUATION_DATA_DIR / "basic_evaluation_results.json"
    summary_output_path = EVALUATION_DATA_DIR / "basic_evaluation_summary.json"

    save_json(results, results_output_path)
    save_json(summary, summary_output_path)

    logger.info(
        "Saved evaluation results for %d question(s) to %s",
        len(results),
        results_output_path,
    )
    logger.info("Saved evaluation summary to %s", summary_output_path)


if __name__ == "__main__":
    main()
