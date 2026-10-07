"""
Standalone LangSmith tracing check for the real RAG chain.

Runs run_rag_query() for exactly one question so the resulting LangSmith
trace can be inspected end-to-end (retriever -> documents -> prompt ->
LLM -> answer). Kept separate from scripts/13_test_rag_chain.py, which
runs multiple sample questions and saves a preview file — this script is
only for tracing inspection and writes nothing to disk.

Requires the same .env vars as scripts/00_test_langsmith_trace.py, plus
OPENAI_API_KEY, PINECONE_API_KEY, and COHERE_API_KEY (the real chain
calls the live Pinecone index and Cohere reranker).
"""

import sys

from dotenv import load_dotenv

from src.config.settings import PROJECT_ROOT
from src.generation.rag_chain import run_rag_query
from src.utils.logger import get_logger


logger = get_logger(__name__)

QUESTION = "What was Infosys' total revenue in fiscal 2025?"
EXPECTED_SOURCE_PAGES = [10, 15]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(PROJECT_ROOT / ".env")

    logger.info("Running RAG chain for question: %s", QUESTION)

    result = run_rag_query(QUESTION)

    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Expected source pages: {EXPECTED_SOURCE_PAGES}")
    print(f"Actual sources: {result['sources']}")


if __name__ == "__main__":
    main()
