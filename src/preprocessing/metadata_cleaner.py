"""
Document metadata cleaning utilities.

This module compacts chunk metadata before embedding and vector database
storage. It does not clean text, chunk documents, create embeddings, save JSON,
or call vector databases.
"""

from langchain_core.documents import Document

from src.utils.logger import get_logger


logger = get_logger(__name__)


ALLOWED_METADATA_KEYS = {
    "source",
    "page",
    "page_number",
    "page_id",
    "chunk_id",
    "chunk_index",
    "title",
    "author",
    "text_length",
    "image_count",
    "drawing_count",
    "width",
    "height",
    "has_low_text",
    "total_pages",
}

REQUIRED_METADATA_KEYS = {
    "chunk_id",
    "page_id",
    "page_number",
}


def clean_document_metadata(
    documents: list[Document],
) -> list[Document]:
    """Keep only compact, useful metadata on chunk Documents.

    Args:
        documents: Chunked LangChain Documents.

    Returns:
        New LangChain Documents with page content preserved and metadata
        reduced to allowed fields.

    Raises:
        ValueError: If documents is empty or required traceability metadata is
            missing.
    """
    try:
        logger.info("Cleaning metadata for %d document(s)", len(documents))

        if not documents:
            raise ValueError("Cannot clean metadata for an empty document list.")

        cleaned_documents: list[Document] = []

        for document in documents:
            missing_required_keys = [
                key
                for key in REQUIRED_METADATA_KEYS
                if key not in document.metadata
            ]

            if missing_required_keys:
                raise ValueError(
                    "Document is missing required metadata key(s): "
                    f"{', '.join(sorted(missing_required_keys))}"
                )

            cleaned_metadata = {
                key: value
                for key, value in document.metadata.items()
                if key in ALLOWED_METADATA_KEYS
            }

            document_kwargs = {
                "page_content": document.page_content,
                "metadata": cleaned_metadata,
            }

            document_id = getattr(document, "id", None)
            if document_id is not None:
                document_kwargs["id"] = document_id

            cleaned_documents.append(Document(**document_kwargs))

        logger.info(
            "Cleaned metadata for %d document(s)",
            len(cleaned_documents),
        )

        return cleaned_documents

    except Exception:
        logger.exception("Failed to clean document metadata")
        raise
