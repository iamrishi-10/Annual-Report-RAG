"""
Multi-question debug sweep, extending 00c_debug_manufacturing_revenue.py's
stage-by-stage approach to a batch of known questions/expected pages.

Calls each pipeline stage directly (multiquery retriever -> Cohere rerank ->
context builder -> answer generator) instead of going through run_rag_query(),
so the expected page's presence/absence can be checked at every stage for
each question, independent of the LangSmith UI. Prints a per-question
breakdown plus a final pass/fail summary. Writes nothing to disk.
"""

import sys

from dotenv import load_dotenv

from src.config.settings import PROJECT_ROOT
from src.context.context_builder import build_context
from src.generation.answer_generator import generate_answer
from src.reranking.cohere_reranker import rerank_documents
from src.retrieval.multiquery_retriever import get_multiquery_retriever
from src.utils.logger import get_logger


logger = get_logger(__name__)

TEST_CASES = [
    {
        "QUESTION": "What was Infosys' total revenue in fiscal 2025?",
        "EXPECTED_SOURCE_PAGE": 10,
    },
    {
        "QUESTION": "What was Infosys' profit after tax in fiscal 2025?",
        "EXPECTED_SOURCE_PAGE": 276,
    },
    {
        "QUESTION": "What was Infosys' total income in fiscal 2025?",
        "EXPECTED_SOURCE_PAGE": 276,
    },
    {
        "QUESTION": "What was Infosys' free cash flow growth in fiscal 2025?",
        "EXPECTED_SOURCE_PAGE": 15,
    },
    {
        "QUESTION": "What is Infosys' capital allocation policy?",
        "EXPECTED_SOURCE_PAGE": 33,
    },
    {
        "QUESTION": "What was Manufacturing revenue from operations in 2025?",
        "EXPECTED_SOURCE_PAGE": 345,
    },
    {
        "QUESTION": "What was Financial Services revenue from operations in 2025?",
        "EXPECTED_SOURCE_PAGE": 345,
    },
    {
        "QUESTION": "What was Retail revenue from operations in 2025?",
        "EXPECTED_SOURCE_PAGE": 345,
    },
    {
        "QUESTION": "What was Communication revenue from operations in 2025?",
        "EXPECTED_SOURCE_PAGE": 345,
    },
    {
        "QUESTION": "What AI capabilities or AI-first initiatives does Infosys mention in the annual report?",
        "EXPECTED_SOURCE_PAGE": 17,
    },
    {
        "QUESTION": "What did the CEO highlight about Infosys' performance in fiscal 2025?",
        "EXPECTED_SOURCE_PAGE": 28,
    },
    {
        "QUESTION": "Who are the members of Infosys' Board of Directors?",
        "EXPECTED_SOURCE_PAGE": 196,
    },
    {
        "QUESTION": "What sustainability or ESG initiatives does Infosys mention for fiscal 2025?",
        "EXPECTED_SOURCE_PAGE": 73,
    },
    {
        "QUESTION": "What are the key risks discussed by Infosys in the annual report?",
        "EXPECTED_SOURCE_PAGE": 224,
    },
    {
        "QUESTION": "Where is the consolidated statement of profit and loss reported?",
        "EXPECTED_SOURCE_PAGE": 304,
    },
    {
        "QUESTION": "Where is the standalone statement of profit and loss reported?",
        "EXPECTED_SOURCE_PAGE": 285,
    },
]


def _pages(documents) -> list:
    return [document.metadata.get("page_number") for document in documents]


def _run_test_case(retriever, question: str, expected_page: int) -> dict:
    retrieved_documents = retriever.invoke(question)
    retrieved_pages = _pages(retrieved_documents)
    retrieval_hit = expected_page in retrieved_pages

    final_documents = rerank_documents(question, retrieved_documents)
    final_pages = _pages(final_documents)
    rerank_hit = expected_page in final_pages

    context = build_context(final_documents)
    context_hit = f"Page: {expected_page}" in context

    answer = generate_answer(question, context)

    return {
        "question": question,
        "expected_page": expected_page,
        "retrieval_hit": retrieval_hit,
        "rerank_hit": rerank_hit,
        "context_hit": context_hit,
        "answer": answer,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(PROJECT_ROOT / ".env")

    retriever = get_multiquery_retriever()

    results = []

    for test_case in TEST_CASES:
        question = test_case["QUESTION"]
        expected_page = test_case["EXPECTED_SOURCE_PAGE"]

        logger.info("Running test case: %s", question)

        result = _run_test_case(retriever, question, expected_page)
        results.append(result)

        print(f"\n=== {question} ===")
        print(f"Expected page: {expected_page}")
        print(f"Retrieval hit: {result['retrieval_hit']}")
        print(f"Rerank hit: {result['rerank_hit']}")
        print(f"Context hit: {result['context_hit']}")
        print(f"Answer: {result['answer']}")

    print("\n\n=== Summary ===")
    print(f"{'Question':<70} {'Retr':<6} {'Rerank':<7} {'Ctx':<6}")
    for result in results:
        print(
            f"{result['question'][:68]:<70} "
            f"{str(result['retrieval_hit']):<6} "
            f"{str(result['rerank_hit']):<7} "
            f"{str(result['context_hit']):<6}"
        )

    total = len(results)
    retrieval_hits = sum(1 for result in results if result["retrieval_hit"])
    rerank_hits = sum(1 for result in results if result["rerank_hit"])
    context_hits = sum(1 for result in results if result["context_hit"])

    print(f"\nRetrieval hit rate: {retrieval_hits}/{total}")
    print(f"Rerank hit rate: {rerank_hits}/{total}")
    print(f"Context hit rate: {context_hits}/{total}")


if __name__ == "__main__":
    main()
