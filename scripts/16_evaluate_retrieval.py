"""
Score retrieval ranking quality from an existing basic evaluation run.

This reuses actual_source_pages / expected_source_pages already saved in
data/evaluation/basic_evaluation_results.json, so it does not call the RAG
chain, retriever, reranker, or LLM again.
"""

import json

from src.config.settings import EVALUATION_DATA_DIR
from src.evaluation.retrieval_evaluator import calculate_retrieval_metrics
from src.utils.json_writer import save_json
from src.utils.logger import get_logger


logger = get_logger(__name__)


def _summarize_retrieval_group(results: list[dict]) -> dict:
    """Build the shared retrieval-metric averages for a group of results."""
    total_questions = len(results)
    retrieval_hit_count = sum(1 for result in results if result["retrieval_hit"])

    return {
        "total_questions": total_questions,
        "retrieval_hit_rate": retrieval_hit_count / total_questions,
        "average_precision_at_k": (
            sum(result["precision_at_k"] for result in results) / total_questions
        ),
        "average_recall_at_k": (
            sum(result["recall_at_k"] for result in results) / total_questions
        ),
        "average_mrr": sum(result["mrr"] for result in results) / total_questions,
    }


def main() -> None:
    """Load basic evaluation results, score retrieval, and save the report."""
    basic_results_path = EVALUATION_DATA_DIR / "basic_evaluation_results.json"

    logger.info("Loading basic evaluation results from %s", basic_results_path)

    with basic_results_path.open("r", encoding="utf-8") as file:
        basic_results = json.load(file)

    retrieval_results = []

    for result in basic_results:
        retrieval_metrics = calculate_retrieval_metrics(
            actual_source_pages=result["actual_source_pages"],
            expected_source_pages=result["expected_source_pages"],
        )

        retrieval_results.append({**result, **retrieval_metrics})

    summary = _summarize_retrieval_group(retrieval_results)

    results_by_category: dict[str, list[dict]] = {}
    for result in retrieval_results:
        results_by_category.setdefault(result["category"], []).append(result)

    summary["category_breakdown"] = {
        category: _summarize_retrieval_group(category_results)
        for category, category_results in results_by_category.items()
    }

    results_output_path = EVALUATION_DATA_DIR / "retrieval_evaluation_results.json"
    summary_output_path = EVALUATION_DATA_DIR / "retrieval_evaluation_summary.json"

    save_json(retrieval_results, results_output_path)
    save_json(summary, summary_output_path)

    logger.info(
        "Saved retrieval evaluation results for %d question(s) to %s",
        len(retrieval_results),
        results_output_path,
    )
    logger.info("Saved retrieval evaluation summary to %s", summary_output_path)


if __name__ == "__main__":
    main()
