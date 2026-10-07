"""
Compare retrieval strategies against the golden questions.

Runs the same golden questions through different retrieval setups and
scores each with calculate_retrieval_metrics(), so we can decide whether
MultiQuery/reranking actually help before tuning the main RAG chain. Does
not change src/generation/rag_chain.py or any retrieval module — this
script only calls existing retrieval/reranking pieces directly for
comparison.
"""

import time

from dotenv import load_dotenv
from langchain_core.documents import Document

from src.config.settings import (
    EVALUATION_DATA_DIR,
    FETCH_K,
    FINAL_TOP_K,
    LAMBDA_MULT,
    PROJECT_ROOT,
    TOP_K,
)
from src.evaluation.basic_evaluator import load_golden_questions
from src.evaluation.retrieval_evaluator import calculate_retrieval_metrics
from src.reranking.cohere_reranker import rerank_documents
from src.retrieval.multiquery_retriever import get_multiquery_retriever
from src.retrieval.retriever import get_retriever
from src.utils.json_writer import save_json
from src.utils.logger import get_logger
from src.vectorstore.vectorstore_indexer import get_vector_store


logger = get_logger(__name__)

MAX_QUESTIONS = 30

# The Cohere trial key is capped at 10 calls/minute. This script calls
# rerank_documents() once per question for every *_rerank strategy, so
# calls are throttled to stay under that cap.
COHERE_MIN_SECONDS_BETWEEN_CALLS = 6.5

_last_cohere_call_time: float | None = None


def _throttled_rerank_documents(question: str, documents: list[Document]) -> list[Document]:
    """Call rerank_documents(), waiting first if the last call was too recent."""
    global _last_cohere_call_time

    if _last_cohere_call_time is not None:
        elapsed = time.monotonic() - _last_cohere_call_time
        wait_seconds = COHERE_MIN_SECONDS_BETWEEN_CALLS - elapsed

        if wait_seconds > 0:
            time.sleep(wait_seconds)

    try:
        return rerank_documents(question, documents)
    finally:
        _last_cohere_call_time = time.monotonic()


def _page_numbers(documents: list[Document]) -> list[int | float]:
    """Extract page_number from each document's metadata, in rank order."""
    return [document.metadata.get("page_number") for document in documents]


def run_similarity_only(question: str) -> list[int | float]:
    """Retrieve with plain similarity search only, no rerank."""
    retriever = get_retriever()
    documents = retriever.invoke(question)[:FINAL_TOP_K]

    return _page_numbers(documents)


def run_similarity_rerank(question: str) -> list[int | float]:
    """Retrieve with plain similarity search, then Cohere rerank.

    All TOP_K candidates are passed to the reranker (not pre-sliced to
    FINAL_TOP_K), since reranking needs a full candidate pool to reorder.
    """
    retriever = get_retriever()
    candidate_documents = retriever.invoke(question)
    final_documents = _throttled_rerank_documents(question, candidate_documents)

    return _page_numbers(final_documents)


def run_multiquery_only(question: str) -> list[int | float]:
    """Retrieve with MultiQueryRetriever only, no rerank."""
    retriever = get_multiquery_retriever()
    documents = retriever.invoke(question)[:FINAL_TOP_K]

    return _page_numbers(documents)


def run_multiquery_rerank(question: str) -> list[int | float]:
    """Retrieve with MultiQueryRetriever, then Cohere rerank (current main chain)."""
    retriever = get_multiquery_retriever()
    candidate_documents = retriever.invoke(question)
    final_documents = _throttled_rerank_documents(question, candidate_documents)

    return _page_numbers(final_documents)


def _get_mmr_retriever():
    """Build an MMR retriever directly from the vector store.

    Not built via get_retriever(), since that is locked to SEARCH_TYPE
    ("similarity") from settings.
    """
    vector_store = get_vector_store()

    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": FETCH_K, "lambda_mult": LAMBDA_MULT},
    )


def run_mmr_only(question: str) -> list[int | float]:
    """Retrieve with MMR search only, no rerank."""
    retriever = _get_mmr_retriever()
    documents = retriever.invoke(question)[:FINAL_TOP_K]

    return _page_numbers(documents)


def run_mmr_rerank(question: str) -> list[int | float]:
    """Retrieve with MMR search, then Cohere rerank.

    All candidates are passed to the reranker (not pre-sliced to
    FINAL_TOP_K), same reasoning as run_similarity_rerank().
    """
    retriever = _get_mmr_retriever()
    candidate_documents = retriever.invoke(question)
    final_documents = _throttled_rerank_documents(question, candidate_documents)

    return _page_numbers(final_documents)


STRATEGIES = {
    "similarity_only": run_similarity_only,
    "similarity_rerank": run_similarity_rerank,
    "multiquery_only": run_multiquery_only,
    "multiquery_rerank": run_multiquery_rerank,
    "mmr_only": run_mmr_only,
    "mmr_rerank": run_mmr_rerank,
}


def _evaluate_strategy(strategy_name: str, strategy_fn, golden_questions: list[dict]) -> dict:
    """Run one strategy over all golden questions and summarize its retrieval metrics."""
    per_question_metrics = []
    failed_question_ids = []

    for golden_item in golden_questions:
        logger.info(
            "[%s] Evaluating question id=%s", strategy_name, golden_item["id"]
        )

        actual_source_pages = strategy_fn(golden_item["question"])
        metrics = calculate_retrieval_metrics(
            actual_source_pages=actual_source_pages,
            expected_source_pages=golden_item["expected_source_pages"],
        )

        per_question_metrics.append(metrics)

        if not metrics["retrieval_hit"]:
            failed_question_ids.append(golden_item["id"])

    total_questions = len(per_question_metrics)
    retrieval_hit_count = sum(1 for metrics in per_question_metrics if metrics["retrieval_hit"])

    return {
        "strategy_name": strategy_name,
        "total_questions": total_questions,
        "retrieval_hit_rate": retrieval_hit_count / total_questions,
        "average_precision_at_k": (
            sum(metrics["precision_at_k"] for metrics in per_question_metrics) / total_questions
        ),
        "average_recall_at_k": (
            sum(metrics["recall_at_k"] for metrics in per_question_metrics) / total_questions
        ),
        "average_mrr": (
            sum(metrics["mrr"] for metrics in per_question_metrics) / total_questions
        ),
        "failed_question_ids": failed_question_ids,
    }


def main() -> None:
    """Run each retrieval strategy over the golden questions and save the comparison."""
    load_dotenv(PROJECT_ROOT / ".env")

    golden_questions = load_golden_questions()[:MAX_QUESTIONS]

    comparison_results = [
        _evaluate_strategy(strategy_name, strategy_fn, golden_questions)
        for strategy_name, strategy_fn in STRATEGIES.items()
    ]

    output_path = EVALUATION_DATA_DIR / "retrieval_strategy_comparison.json"
    save_json(comparison_results, output_path)

    logger.info("Saved retrieval strategy comparison to %s", output_path)


if __name__ == "__main__":
    main()
