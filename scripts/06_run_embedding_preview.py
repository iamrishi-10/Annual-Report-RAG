"""
Run the full preprocessing pipeline and save a small embedding preview.
"""

from dotenv import load_dotenv

from src.chunking.chunker import chunk_documents
from src.config.settings import (
    EMBEDDING_DIMENSION,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
)
from src.document_loading.langchain_loader import load_with_langchain
from src.document_loading.pdf_loader import inspect_pdf_with_fitz
from src.embeddings.embedder import embed_documents
from src.preprocessing.metadata_cleaner import clean_document_metadata
from src.preprocessing.metadata_enricher import (
    enrich_documents_with_pdf_inspection,
)
from src.utils.json_writer import save_json
from src.utils.logger import get_logger


logger = get_logger(__name__)

PREVIEW_CHUNK_COUNT = 5
EMBEDDING_PREVIEW_VALUES = 10


def main() -> None:
    """Embed a small cleaned chunk preview and save validation details."""
    load_dotenv(PROJECT_ROOT / ".env")

    documents = load_with_langchain()
    pdf_inspection = inspect_pdf_with_fitz()
    enriched_documents = enrich_documents_with_pdf_inspection(
        documents,
        pdf_inspection,
    )
    chunked_documents = chunk_documents(enriched_documents)
    cleaned_documents = clean_document_metadata(chunked_documents)
    preview_documents = cleaned_documents[:PREVIEW_CHUNK_COUNT]
    embeddings = embed_documents(preview_documents)

    embedding_preview = []

    for document, embedding in zip(preview_documents, embeddings):
        embedding_dimension = len(embedding)

        if embedding_dimension != EMBEDDING_DIMENSION:
            raise ValueError(
                "Embedding dimension does not match settings: "
                f"actual={embedding_dimension} expected={EMBEDDING_DIMENSION}"
            )

        embedding_preview.append(
            {
                "chunk_id": document.metadata["chunk_id"],
                "page_number": document.metadata["page_number"],
                "chunk_length": len(document.page_content.strip()),
                "embedding_dimension": embedding_dimension,
                "embedding_preview": embedding[:EMBEDDING_PREVIEW_VALUES],
            }
        )

    output_path = PROCESSED_DATA_DIR / "embedding_preview.json"

    save_json(embedding_preview, output_path)

    logger.info(
        "Saved embedding preview for %d chunk(s) to %s",
        len(embedding_preview),
        output_path,
    )


if __name__ == "__main__":
    main()
