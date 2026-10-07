"""
Pinecone vector indexing utilities.

This module embeds already-cleaned chunk Documents and upserts them into
Pinecone via LangChain's PineconeVectorStore. It does not load PDFs, inspect
PDFs, enrich metadata, chunk documents, clean metadata, perform retrieval, or
call the LLM.
"""

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from src.config.settings import PINECONE_NAMESPACE
from src.embeddings.embedding_model import get_embedding_model
from src.utils.logger import get_logger
from src.vectorstore.pinecone_client import get_pinecone_index


logger = get_logger(__name__)

FETCH_BATCH_SIZE = 100


def get_vector_store() -> PineconeVectorStore:
    """Build a PineconeVectorStore over the existing Pinecone index.

    Shared by both indexing and retrieval so the index/embedding/namespace
    wiring is defined in exactly one place.

    Returns:
        A PineconeVectorStore scoped to PINECONE_NAMESPACE.

    Raises:
        ValueError: If PINECONE_NAMESPACE is empty.
    """
    try:
        if not PINECONE_NAMESPACE.strip():
            raise ValueError("PINECONE_NAMESPACE must not be empty.")

        index = get_pinecone_index()
        embedding_model = get_embedding_model()

        vector_store = PineconeVectorStore(
            index=index,
            embedding=embedding_model,
            namespace=PINECONE_NAMESPACE,
        )

        return vector_store

    except Exception:
        logger.exception("Failed to build Pinecone vector store")
        raise


def get_existing_chunk_ids(chunk_ids: list[str]) -> set[str]:
    """Return the subset of chunk_ids already present in the Pinecone index.

    Args:
        chunk_ids: Chunk IDs to check for existing vectors.

    Returns:
        The chunk_ids that already have a vector in the configured namespace.

    Raises:
        ValueError: If chunk_ids is empty or PINECONE_NAMESPACE is empty.
    """
    try:
        if not chunk_ids:
            raise ValueError("chunk_ids must not be empty.")

        if not PINECONE_NAMESPACE.strip():
            raise ValueError("PINECONE_NAMESPACE must not be empty.")

        index = get_pinecone_index()

        existing_chunk_ids: set[str] = set()

        for batch_start in range(0, len(chunk_ids), FETCH_BATCH_SIZE):
            batch_ids = chunk_ids[batch_start : batch_start + FETCH_BATCH_SIZE]

            response = index.fetch(ids=batch_ids, namespace=PINECONE_NAMESPACE)

            existing_chunk_ids.update(response.vectors.keys())

        logger.info(
            "Found %d of %d chunk(s) already indexed in Pinecone",
            len(existing_chunk_ids),
            len(chunk_ids),
        )

        return existing_chunk_ids

    except Exception:
        logger.exception("Failed to check existing chunk ids in Pinecone")
        raise


def index_documents_to_pinecone(
    documents: list[Document],
) -> list[str]:
    """Embed cleaned chunk Documents and upsert them into Pinecone.

    Args:
        documents: Cleaned chunk Documents. Each document must include
            chunk_id in its metadata and non-empty page_content.

    Returns:
        The chunk_id values that were indexed, in input order.

    Raises:
        ValueError: If documents is empty, any document is missing chunk_id,
            any document has empty page_content, or PINECONE_NAMESPACE is
            empty.
    """
    try:
        logger.info("Indexing %d document(s) to Pinecone", len(documents))

        if not documents:
            raise ValueError("Cannot index an empty document list.")

        if not PINECONE_NAMESPACE.strip():
            raise ValueError("PINECONE_NAMESPACE must not be empty.")

        chunk_ids: list[str] = []

        for document_index, document in enumerate(documents):
            chunk_id = document.metadata.get("chunk_id")

            if not chunk_id:
                raise ValueError(
                    f"Document at index {document_index} is missing chunk_id."
                )

            if not document.page_content.strip():
                raise ValueError(
                    f"Document at index {document_index} has empty page_content."
                )

            chunk_ids.append(chunk_id)

        vector_store = get_vector_store()

        logger.info(
            "Upserting %d document(s) to Pinecone namespace %s",
            len(documents),
            PINECONE_NAMESPACE,
        )

        indexed_ids = vector_store.add_documents(documents, ids=chunk_ids)

        logger.info(
            "Indexed %d document(s) to Pinecone namespace %s",
            len(indexed_ids),
            PINECONE_NAMESPACE,
        )

        return indexed_ids

    except Exception:
        logger.exception("Failed to index documents to Pinecone")
        raise
