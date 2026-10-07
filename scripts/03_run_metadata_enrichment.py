"""
Run metadata enrichment and save a small JSON preview.
"""

from src.config.settings import PROCESSED_DATA_DIR
from src.document_loading.langchain_loader import load_with_langchain
from src.document_loading.pdf_loader import inspect_pdf_with_fitz
from src.preprocessing.metadata_enricher import (
    enrich_documents_with_pdf_inspection,
)
from src.utils.json_writer import save_json
from src.utils.logger import get_logger


logger = get_logger(__name__)

PREVIEW_DOCUMENT_COUNT = 5
PAGE_CONTENT_PREVIEW_LENGTH = 120


def main() -> None:
    """Enrich PDF page documents and save a small metadata preview."""
    documents = load_with_langchain()
    pdf_inspection = inspect_pdf_with_fitz()
    enriched_documents = enrich_documents_with_pdf_inspection(
        documents,
        pdf_inspection,
    )

    metadata_preview = [
        {
            "page_content_preview": document.page_content.strip()[
                :PAGE_CONTENT_PREVIEW_LENGTH
            ],
            "metadata": document.metadata,
        }
        for document in enriched_documents[:PREVIEW_DOCUMENT_COUNT]
    ]
    output_path = PROCESSED_DATA_DIR / "enriched_metadata_preview.json"

    save_json(metadata_preview, output_path)

    logger.info(
        "Saved metadata enrichment preview for %d document(s) to %s",
        len(metadata_preview),
        output_path,
    )


if __name__ == "__main__":
    main()
