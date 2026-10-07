"""
Score the permanent RAG system (similarity -> MultiQuery -> Cohere rerank ->
answer generation) with RAGAS.

Reuses the existing basic evaluation results (no RAG re-call) via
ragas_evaluator.py, then calls RAGAS's faithfulness, answer_relevancy,
context_precision, context_recall, and answer_correctness metrics.

This is the first script that calls RAGAS, and it costs real OpenAI usage:
each metric makes its own LLM (and, for answer_relevancy/answer_correctness,
embedding) calls per question, on top of whatever scripts/15 already spent
generating the underlying answers.
"""

import math

from dotenv import load_dotenv
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.config.settings import EVALUATION_DATA_DIR, PROJECT_ROOT
from src.embeddings.embedding_model import get_embedding_model
from src.evaluation.ragas_evaluator import (
    build_ragas_dataset,
    load_basic_evaluation_results,
)
from src.generation.llm_client import get_llm
from src.utils.json_writer import save_json
from src.utils.logger import get_logger


logger = get_logger(__name__)

METRICS = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
]


def _average_metric_scores(per_question_results: list[dict]) -> dict:
    """Average each metric across questions, skipping missing/NaN scores."""
    metric_names = [metric.name for metric in METRICS]
    averages = {}

    for metric_name in metric_names:
        scores = [
            result[metric_name]
            for result in per_question_results
            if metric_name in result
            and result[metric_name] is not None
            and not math.isnan(result[metric_name])
        ]

        averages[metric_name] = sum(scores) / len(scores) if scores else None

    return averages


def main() -> None:
    """Run RAGAS metrics over the saved basic evaluation results and save the report."""
    load_dotenv(PROJECT_ROOT / ".env")

    evaluation_results = load_basic_evaluation_results()
    ragas_dataset = build_ragas_dataset(evaluation_results)

    llm = LangchainLLMWrapper(get_llm())
    embeddings = LangchainEmbeddingsWrapper(get_embedding_model())

    logger.info(
        "Running RAGAS metrics %s over %d question(s)",
        [metric.name for metric in METRICS],
        len(evaluation_results),
    )

    ragas_result = evaluate(
        dataset=ragas_dataset,
        metrics=METRICS,
        llm=llm,
        embeddings=embeddings,
    )

    per_question_scores = ragas_result.scores

    if len(per_question_scores) != len(evaluation_results):
        raise ValueError(
            "RAGAS score count does not match evaluation result count: "
            f"scores={len(per_question_scores)} "
            f"results={len(evaluation_results)}"
        )

    per_question_results = [
        {
            "id": evaluation_result["id"],
            "question": evaluation_result["question"],
            **scores,
        }
        for evaluation_result, scores in zip(evaluation_results, per_question_scores)
    ]

    summary = {
        "total_questions": len(per_question_results),
        "average_scores": _average_metric_scores(per_question_results),
    }

    results_output_path = EVALUATION_DATA_DIR / "ragas_evaluation_results.json"
    summary_output_path = EVALUATION_DATA_DIR / "ragas_evaluation_summary.json"

    save_json(per_question_results, results_output_path)
    save_json(summary, summary_output_path)

    logger.info(
        "Saved RAGAS evaluation results for %d question(s) to %s",
        len(per_question_results),
        results_output_path,
    )
    logger.info("Saved RAGAS evaluation summary to %s", summary_output_path)


if __name__ == "__main__":
    main()
