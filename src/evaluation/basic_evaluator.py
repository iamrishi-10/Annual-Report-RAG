"""
Helper functions for evaluating RAG results.

This module currently only loads and validates the golden evaluation
dataset. It does not run the RAG chain, retrieval, reranking, call the
LLM, generate answers, or save JSON.
"""

import json

from src.config.prompts import NOT_FOUND_MESSAGE
from src.config.settings import EVALUATION_DATA_DIR
from src.utils.logger import get_logger


logger = get_logger(__name__)

REQUIRED_FIELDS = [
    "id",
    "question",
    "category",
    "expected_answer",
    "expected_facts",
    "expected_source_pages",
    "difficulty",
]


def load_golden_questions() -> list[dict]:
    """Load and validate the golden evaluation dataset.

    Returns:
        The list of golden question dictionaries, as loaded from
        data/evaluation/golden_questions.json.

    Raises:
        FileNotFoundError: If the golden dataset file does not exist.
        ValueError: If the JSON is not a list, or any item is missing a
            required field, or expected_facts/expected_source_pages is not
            a list.
    """
    golden_questions_path = EVALUATION_DATA_DIR / "golden_questions.json"

    try:
        logger.info("Loading golden questions from %s", golden_questions_path)

        if not golden_questions_path.exists():
            raise FileNotFoundError(
                f"Golden dataset not found at {golden_questions_path}."
            )

        with golden_questions_path.open("r", encoding="utf-8") as file:
            golden_questions = json.load(file)

        if not isinstance(golden_questions, list):
            raise ValueError(
                "Golden dataset must be a JSON list of question objects."
            )

        for question_index, question_item in enumerate(golden_questions):
            missing_fields = [
                field for field in REQUIRED_FIELDS if field not in question_item
            ]

            if missing_fields:
                raise ValueError(
                    f"Golden question at index {question_index} "
                    f"(id={question_item.get('id')!r}) is missing required "
                    f"field(s): {missing_fields}."
                )

            if not isinstance(question_item["expected_facts"], list):
                raise ValueError(
                    f"Golden question at index {question_index} "
                    f"(id={question_item.get('id')!r}) has a non-list "
                    f"expected_facts field."
                )

            if not isinstance(question_item["expected_source_pages"], list):
                raise ValueError(
                    f"Golden question at index {question_index} "
                    f"(id={question_item.get('id')!r}) has a non-list "
                    f"expected_source_pages field."
                )

        logger.info("Loaded and validated %d golden question(s)", len(golden_questions))

        return golden_questions

    except Exception:
        logger.exception("Failed to load golden questions")
        raise


def calculate_source_page_hit(
    actual_source_pages: list[int | float],
    expected_source_pages: list[int],
) -> bool:
    """Check whether at least one expected source page was actually retrieved.

    Page numbers are normalized with int() before comparing, since actual
    source pages (e.g. from Pinecone metadata) may come back as floats
    (10.0) while expected_source_pages are plain ints (10).

    Args:
        actual_source_pages: Page numbers from the RAG chain's sources.
        expected_source_pages: Page numbers the golden question expects.

    Returns:
        True if at least one expected page appears in the actual pages,
        False otherwise.
    """
    normalized_actual_pages = {int(page) for page in actual_source_pages}
    normalized_expected_pages = {int(page) for page in expected_source_pages}

    return not normalized_actual_pages.isdisjoint(normalized_expected_pages)


def evaluate_single_result(
    golden_item: dict,
    rag_result: dict,
) -> dict:
    """Compare one golden question with one RAG chain result.

    Args:
        golden_item: One golden question dict (see load_golden_questions()).
        rag_result: One run_rag_query() result dict (answer, sources,
            retrieved_chunk_count, final_context_chunk_count).

    Returns:
        A dict with the golden question's expected fields, the RAG
        result's actual fields, and the calculated source_page_hit metric.
        answered is False both when the answer is empty and when it is the
        fixed fallback message (NOT_FOUND_MESSAGE), since a fallback is not
        a successful answer.
    """
    actual_source_pages = [
        source.get("page_number") for source in rag_result.get("sources", [])
    ]

    source_page_hit = calculate_source_page_hit(
        actual_source_pages=actual_source_pages,
        expected_source_pages=golden_item["expected_source_pages"],
    )

    actual_answer = rag_result.get("answer", "")
    answered = bool(actual_answer.strip()) and actual_answer.strip() != NOT_FOUND_MESSAGE

    return {
        "id": golden_item["id"],
        "question": golden_item["question"],
        "category": golden_item["category"],
        "difficulty": golden_item["difficulty"],
        "expected_answer": golden_item["expected_answer"],
        "expected_source_pages": golden_item["expected_source_pages"],
        "actual_answer": rag_result.get("answer", ""),
        "actual_source_pages": actual_source_pages,
        "actual_sources": rag_result.get("sources", []),
        "contexts": rag_result.get("contexts", []),
        "retrieved_chunk_count": rag_result.get("retrieved_chunk_count", 0),
        "final_context_chunk_count": rag_result.get("final_context_chunk_count", 0),
        "source_page_hit": source_page_hit,
        "answered": answered,
    }


def _summarize_group(results: list[dict]) -> dict:
    """Build the shared count/rate/average fields for a group of results."""
    total_questions = len(results)
    answered_count = sum(1 for result in results if result["answered"])
    source_page_hit_count = sum(1 for result in results if result["source_page_hit"])

    return {
        "total_questions": total_questions,
        "answered_count": answered_count,
        "source_page_hit_rate": source_page_hit_count / total_questions,
    }


def summarize_evaluation_results(results: list[dict]) -> dict:
    """Summarize a list of evaluate_single_result() outputs into one report.

    Args:
        results: A list of dicts, each produced by evaluate_single_result().

    Returns:
        A dict with total_questions, answered_count, failed_count,
        source_page_hit_rate, average_retrieved_chunk_count,
        average_final_context_chunk_count, and a category_breakdown
        grouping the same core metrics by category.
    """
    if not results:
        return {
            "total_questions": 0,
            "answered_count": 0,
            "failed_count": 0,
            "source_page_hit_rate": 0.0,
            "average_retrieved_chunk_count": 0.0,
            "average_final_context_chunk_count": 0.0,
            "category_breakdown": {},
        }

    total_questions = len(results)
    answered_count = sum(1 for result in results if result["answered"])

    results_by_category: dict[str, list[dict]] = {}
    for result in results:
        results_by_category.setdefault(result["category"], []).append(result)

    category_breakdown = {
        category: _summarize_group(category_results)
        for category, category_results in results_by_category.items()
    }

    return {
        "total_questions": total_questions,
        "answered_count": answered_count,
        "failed_count": total_questions - answered_count,
        "source_page_hit_rate": (
            sum(1 for result in results if result["source_page_hit"]) / total_questions
        ),
        "average_retrieved_chunk_count": (
            sum(result["retrieved_chunk_count"] for result in results) / total_questions
        ),
        "average_final_context_chunk_count": (
            sum(result["final_context_chunk_count"] for result in results) / total_questions
        ),
        "category_breakdown": category_breakdown,
    }
