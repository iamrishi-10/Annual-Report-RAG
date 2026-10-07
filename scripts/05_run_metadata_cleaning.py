"""
Run the chunking pipeline, clean chunk metadata, and save a small JSON preview.
"""

from src.chunking.chunker import chunk_documents
from src.config.settings import PROCESSED_DATA_DIR
from src.document_loading.langchain_loader import load_with_langchain
from src.document_loading.pdf_loader import inspect_pdf_with_fitz
from src.preprocessing.metadata_cleaner import clean_document_metadata
from src.preprocessing.metadata_enricher import (
    enrich_documents_with_pdf_inspection,
)
from src.utils.json_writer import save_json
from src.utils.logger import get_logger


logger = get_logger(__name__)

PREVIEW_CHUNK_COUNT = 10
CHUNK_CONTENT_PREVIEW_LENGTH = 200


def main() -> None:
    """Run metadata cleaning pipeline and save a cleaned chunk preview."""
    documents = load_with_langchain()
    pdf_inspection = inspect_pdf_with_fitz()
    enriched_documents = enrich_documents_with_pdf_inspection(
        documents,
        pdf_inspection,
    )
    chunked_documents = chunk_documents(enriched_documents)
    cleaned_documents = clean_document_metadata(chunked_documents)

    cleaned_chunk_preview = [
        {
            "chunk_content_preview": document.page_content.strip()[
                :CHUNK_CONTENT_PREVIEW_LENGTH
            ],
            "chunk_length": len(document.page_content.strip()),
            "metadata": document.metadata,
        }
        for document in cleaned_documents[:PREVIEW_CHUNK_COUNT]
    ]
    output_path = PROCESSED_DATA_DIR / "cleaned_chunk_preview.json"

    save_json(cleaned_chunk_preview, output_path)

    logger.info(
        "Saved cleaned chunk preview for %d chunk(s) to %s",
        len(cleaned_chunk_preview),
        output_path,
    )


if __name__ == "__main__":
    main()
