"""
RAGAS dataset preparation utilities.

This module converts the existing basic evaluation results into the format
RAGAS expects for scoring answer/context quality. It does not call the RAG
chain, retrieval, reranking, or the LLM, and does not compute any RAGAS
metrics itself.
"""

import json

from ragas import EvaluationDataset

from src.config.settings import EVALUATION_DATA_DIR
from src.utils.logger import get_logger


logger = get_logger(__name__)

REQUIRED_FIELDS = ["question", "actual_answer", "expected_answer", "contexts"]


def load_basic_evaluation_results() -> list[dict]:
    """Load the saved basic evaluation results.

    Returns:
        The list of per-question result dicts saved by
        scripts/15_run_basic_evaluation.py.

    Raises:
        FileNotFoundError: If basic_evaluation_results.json does not exist.
    """
    results_path = EVALUATION_DATA_DIR / "basic_evaluation_results.json"

    try:
        logger.info("Loading basic evaluation results from %s", results_path)

        if not results_path.exists():
            raise FileNotFoundError(
                f"Basic evaluation results not found at {results_path}."
            )

        with results_path.open("r", encoding="utf-8") as file:
            evaluation_results = json.load(file)

        logger.info(
            "Loaded %d basic evaluation result(s)",
            len(evaluation_results),
        )

        return evaluation_results

    except Exception:
        logger.exception("Failed to load basic evaluation results")
        raise


def build_ragas_dataset(evaluation_results: list[dict]) -> EvaluationDataset:
    """Convert basic evaluation results into a RAGAS EvaluationDataset.

    Args:
        evaluation_results: Per-question result dicts, each with question,
            actual_answer, expected_answer, and contexts (see
            evaluate_single_result() in basic_evaluator.py).

    Returns:
        A RAGAS EvaluationDataset built from one SingleTurnSample per
        result: user_input (question), response (actual_answer),
        retrieved_contexts (contexts), reference (expected_answer).

    Raises:
        ValueError: If evaluation_results is empty, any result is missing a
            required field (question, actual_answer, expected_answer,
            contexts), or contexts is not a list.
    """
    try:
        logger.info(
            "Building RAGAS dataset from %d evaluation result(s)",
            len(evaluation_results),
        )

        if not evaluation_results:
            raise ValueError("Cannot build a RAGAS dataset from an empty result list.")

        for index, result in enumerate(evaluation_results):
            missing_fields = [
                field for field in REQUIRED_FIELDS if field not in result
            ]

            if missing_fields:
                raise ValueError(
                    f"Result at index {index} is missing required fields: {missing_fields}"
                )

            if not isinstance(result["contexts"], list):
                raise ValueError(
                    f"Result at index {index} must contain contexts as a list."
                )

        samples = [
            {
                "user_input": result["question"],
                "response": result["actual_answer"],
                "retrieved_contexts": result["contexts"],
                "reference": result["expected_answer"],
            }
            for result in evaluation_results
        ]

        ragas_dataset = EvaluationDataset.from_list(samples)

        logger.info("Built RAGAS dataset with %d sample(s)", len(samples))

        return ragas_dataset

    except Exception:
        logger.exception("Failed to build RAGAS dataset")
        raise
