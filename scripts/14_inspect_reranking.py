"""
Inspect Cohere reranking behavior: TOP_K candidates vs final reranked chunks.

Purpose: see exactly which candidates Cohere promotes/demotes, since
rag_chain_preview.json only shows the final sources, not the full
reranking behavior. No LLM call, no prompt, no context building.
"""

from dotenv import load_dotenv

from src.config.settings import PROCESSED_DATA_DIR, PROJECT_ROOT
from src.reranking.cohere_reranker import rerank_documents
from src.retrieval.retriever import get_retriever
from src.utils.json_writer import save_json
from src.utils.logger import get_logger


logger = get_logger(__name__)

CONTENT_PREVIEW_LENGTH = 600

SAMPLE_QUESTIONS = [
    "What was Infosys total revenue in fiscal year 24-2025?",
    "What is the company's approach to AI?",
    "Who is the CEO of Infosys?",
    "What are the key business segments?",
]


def _to_preview(document, position: int, include_relevance_score: bool) -> dict:
    """Build a compact preview dict for one Document at a given rank position."""
    preview = {
        "position": position,
        "page_number": document.metadata.get("page_number"),
        "chunk_id": document.metadata.get("chunk_id"),
    }

    if include_relevance_score:
        preview["relevance_score"] = document.metadata.get("relevance_score")

    preview["content_preview"] = document.page_content.strip()[
        :CONTENT_PREVIEW_LENGTH
    ]

    return preview


def main() -> None:
    """Retrieve candidates, rerank them, and save both sets for inspection."""
    load_dotenv(PROJECT_ROOT / ".env")

    retriever = get_retriever()

    reranking_inspection = []

    for question in SAMPLE_QUESTIONS:
        logger.info("Inspecting reranking for question: %s", question)

        candidate_documents = retriever.invoke(question)
        final_documents = rerank_documents(question, candidate_documents)

        reranking_inspection.append(
            {
                "question": question,
                "candidate_chunk_count": len(candidate_documents),
                "final_reranked_chunk_count": len(final_documents),
                "candidate_chunks": [
                    _to_preview(document, document_index + 1, include_relevance_score=False)
                    for document_index, document in enumerate(candidate_documents)
                ],
                "final_reranked_chunks": [
                    _to_preview(document, document_index + 1, include_relevance_score=True)
                    for document_index, document in enumerate(final_documents)
                ],
            }
        )

    output_path = PROCESSED_DATA_DIR / "reranking_inspection.json"

    save_json(reranking_inspection, output_path)

    logger.info(
        "Saved reranking inspection for %d question(s) to %s",
        len(reranking_inspection),
        output_path,
    )


if __name__ == "__main__":
    main()
