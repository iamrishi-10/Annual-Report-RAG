"""
Document chunking utilities.

This module splits enriched LangChain page Documents into smaller chunks and
assigns stable chunk identifiers. It does not clean text, create embeddings,
save JSON, or call vector databases.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import CHUNK_OVERLAP, CHUNK_SIZE
from src.utils.logger import get_logger


logger = get_logger(__name__)


def chunk_documents(
    documents: list[Document],
) -> list[Document]:
    """Split enriched page Documents into chunk Documents.

    Args:
        documents: Enriched LangChain Documents. Each document must include
            page_id in its metadata.

    Returns:
        Chunked LangChain Documents with existing metadata preserved and
        chunk_index and chunk_id added.

    Raises:
        ValueError: If documents is empty, chunk settings are invalid, or a
            document is missing required page_id metadata.
    """
    try:
        logger.info("Starting chunking for %d document(s)", len(documents))

        if not documents:
            raise ValueError("Cannot chunk an empty document list.")

        if CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE must be greater than 0.")

        if CHUNK_OVERLAP < 0:
            raise ValueError("CHUNK_OVERLAP must be greater than or equal to 0.")

        if CHUNK_OVERLAP >= CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunked_documents: list[Document] = []

        for document in documents:
            page_id = document.metadata.get("page_id")
            if not page_id:
                raise ValueError(
                    "Document is missing required 'page_id' metadata."
                )

            page_chunks = splitter.split_documents([document])

            for chunk_index, chunk in enumerate(page_chunks):
                enriched_metadata = {
                    **chunk.metadata,
                    "chunk_index": chunk_index,
                    "chunk_id": f"{page_id}_chunk_{chunk_index}",
                }

                chunked_documents.append(
                    Document(
                        page_content=chunk.page_content,
                        metadata=enriched_metadata,
                    )
                )

        logger.info(
            "Created %d chunk(s) from %d document(s)",
            len(chunked_documents),
            len(documents),
        )

        return chunked_documents

    except Exception:
        logger.exception("Failed to chunk documents")
        raise


if __name__ == "__main__":
    sample_document = Document(
        page_content=(
            "This is sample annual report text for chunking validation. " * 40
        ),
        metadata={
            "source": "sample.pdf",
            "page": 0,
            "page_number": 1,
            "page_id": "sample_page_1",
            "text_length": 2320,
            "image_count": 1,
            "drawing_count": 2,
            "width": 612.0,
            "height": 792.0,
            "has_low_text": False,
        },
    )

    sample_chunks = chunk_documents([sample_document])
    first_chunk = sample_chunks[0]

    logger.info(
        "Chunker smoke test created %d chunk(s)",
        len(sample_chunks),
    )
    logger.info(
        "First chunk id=%s",
        first_chunk.metadata["chunk_id"],
    )
    logger.info(
        "First chunk index=%d",
        first_chunk.metadata["chunk_index"],
    )
    logger.info(
        "First chunk page id=%s",
        first_chunk.metadata["page_id"],
    )
    logger.info(
        "First chunk text length=%d",
        len(first_chunk.page_content),
    )
