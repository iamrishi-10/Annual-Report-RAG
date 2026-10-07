"""
One-off debug script for the "Manufacturing revenue from operations in 2025"
known failure case.

Calls each pipeline stage directly (multiquery retriever -> Cohere rerank ->
context builder -> answer generator) instead of going through run_rag_query(),
so page 345's presence/absence can be checked at every stage in the terminal,
independent of the LangSmith UI.
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

QUESTION = "What was Manufacturing revenue from operations in 2025?"
EXPECTED_SOURCE_PAGE = 345


def _pages(documents) -> list:
    return [document.metadata.get("page_number") for document in documents]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(PROJECT_ROOT / ".env")

    retriever = get_multiquery_retriever()
    retrieved_documents = retriever.invoke(QUESTION)
    retrieved_pages = _pages(retrieved_documents)

    print(f"--- MultiQueryRetriever output ({len(retrieved_documents)} docs) ---")
    print(f"Pages: {retrieved_pages}")
    print(f"Page {EXPECTED_SOURCE_PAGE} present: {EXPECTED_SOURCE_PAGE in retrieved_pages}")

    final_documents = rerank_documents(QUESTION, retrieved_documents)
    final_pages = _pages(final_documents)

    print(f"\n--- cohere_rerank_documents output ({len(final_documents)} docs) ---")
    print(f"Pages: {final_pages}")
    print(f"Page {EXPECTED_SOURCE_PAGE} present: {EXPECTED_SOURCE_PAGE in final_pages}")

    context = build_context(final_documents)

    print(f"\n--- build_rag_context output ---")
    print(f"Contains 'Page: {EXPECTED_SOURCE_PAGE}': {f'Page: {EXPECTED_SOURCE_PAGE}' in context}")
    print(f"Contains 'Manufacturing': {'Manufacturing' in context}")
    print(f"Contains '25,207': {'25,207' in context}")
    print(f"Contains 'revenue from operations' (case-insensitive): {'revenue from operations' in context.lower()}")

    answer = generate_answer(QUESTION, context)

    print(f"\n--- Final ChatOpenAI answer ---")
    print(answer)


if __name__ == "__main__":
    main()
