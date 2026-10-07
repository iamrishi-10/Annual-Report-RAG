"""
Document embedding utilities.

This module embeds cleaned chunk Documents using the configured embedding
model. It does not call Pinecone, save JSON, load PDFs, chunk documents, or
clean metadata.
"""

from langchain_core.documents import Document

from src.embeddings.embedding_model import get_embedding_model
from src.utils.logger import get_logger


logger = get_logger(__name__)


def embed_documents(
    documents: list[Document],
) -> list[list[float]]:
    """Generate embedding vectors for document page content.

    Args:
        documents: Cleaned chunk Documents to embed.

    Returns:
        Embedding vectors aligned with the input document order.

    Raises:
        ValueError: If documents is empty, any document has empty page content,
            or the embedding count does not match the document count.
    """
    try:
        logger.info("Embedding %d document(s)", len(documents))

        if not documents:
            raise ValueError("Cannot embed an empty document list.")

        for document_index, document in enumerate(documents):
            if not document.page_content.strip():
                raise ValueError(
                    f"Document at index {document_index} has empty page_content."
                )

        embedding_model = get_embedding_model()
        document_texts = [
            document.page_content
            for document in documents
        ]
        embeddings = embedding_model.embed_documents(document_texts)

        if len(embeddings) != len(documents):
            raise ValueError(
                "Embedding count does not match document count: "
                f"embeddings={len(embeddings)} documents={len(documents)}"
            )

        logger.info("Created %d embedding vector(s)", len(embeddings))

        return embeddings

    except Exception:
        logger.exception("Failed to embed documents")
        raise
