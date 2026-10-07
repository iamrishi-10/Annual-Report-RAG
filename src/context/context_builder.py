"""
Context formatting utilities.

This module converts already-retrieved LangChain Documents into a single
formatted context string for the LLM prompt. It does not retrieve documents,
call Pinecone, load PDFs, chunk documents, clean metadata, or call the LLM.
"""

from langchain_core.documents import Document
from langsmith import traceable

from src.utils.logger import get_logger


logger = get_logger(__name__)


@traceable(name="build_rag_context")
def build_context(documents: list[Document]) -> str:
    """Format retrieved Documents into a single numbered-source context string.

    Args:
        documents: Retrieved Documents, each with chunk_id and page_number in
            metadata and non-empty page_content.

    Returns:
        A single string with one numbered source block per document, in
        input order.

    Raises:
        ValueError: If documents is empty, any document has empty
            page_content, or any document is missing page_number or
            chunk_id in metadata.
    """
    try:
        logger.info("Building context from %d document(s)", len(documents))

        if not documents:
            raise ValueError("Cannot build context from an empty document list.")

        source_blocks: list[str] = []

        for document_index, document in enumerate(documents):
            if not document.page_content.strip():
                raise ValueError(
                    f"Document at index {document_index} has empty page_content."
                )

            page_number = document.metadata.get("page_number")

            if page_number is None:
                raise ValueError(
                    f"Document at index {document_index} is missing page_number."
                )

            chunk_id = document.metadata.get("chunk_id")

            if not chunk_id:
                raise ValueError(
                    f"Document at index {document_index} is missing chunk_id."
                )

            source_blocks.append(
                f"[Source {document_index + 1}]\n"
                f"Page: {page_number}\n"
                f"Chunk ID: {chunk_id}\n"
                f"Content:\n{document.page_content.strip()}"
            )

        context = "\n\n".join(source_blocks)

        logger.info("Built context from %d document(s)", len(documents))

        return context

    except Exception:
        logger.exception("Failed to build context")
        raise
