"""
Document metadata enrichment utilities.

This module adds page-level PDF inspection metadata to LangChain Document
objects. It does not clean text, chunk documents, create embeddings, or assign
chunk identifiers.
"""

from pathlib import Path

from langchain_core.documents import Document

from src.document_loading.document_models import (
    PDFInspectionResult,
    PageInspection,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


def enrich_documents_with_pdf_inspection(
    documents: list[Document],
    pdf_inspection: PDFInspectionResult,
) -> list[Document]:
    """Add PDF inspection metadata to one LangChain Document per page.

    Args:
        documents: LangChain Documents loaded from the PDF, one per page.
        pdf_inspection: Direct PyMuPDF inspection result for the same PDF.

    Returns:
        New LangChain Documents with page-level inspection metadata added.

    Raises:
        ValueError: If the LangChain document count does not match the PDF
            inspection page count.
    """
    try:
        logger.info(
            "Enriching %d document(s) with PDF inspection metadata",
            len(documents),
        )

        if len(documents) != pdf_inspection.total_pages:
            raise ValueError(
                "Document count does not match PDF inspection page count: "
                f"documents={len(documents)} "
                f"pdf_pages={pdf_inspection.total_pages}"
            )

        if len(pdf_inspection.pages) != pdf_inspection.total_pages:
            raise ValueError(
                "PDF inspection page detail count does not match total pages: "
                f"page_details={len(pdf_inspection.pages)} "
                f"total_pages={pdf_inspection.total_pages}"
            )

        inspection_by_page_number = {
            page.page_number: page
            for page in pdf_inspection.pages
        }
        source_name = Path(pdf_inspection.source_file).stem
        enriched_documents: list[Document] = []

        for document in documents:
            if "page" not in document.metadata:
                raise ValueError(
                    "LangChain document is missing required 'page' metadata."
                )

            langchain_page_number = document.metadata["page"] + 1

            if langchain_page_number not in inspection_by_page_number:
                raise ValueError(
                    f"No PDF inspection found for page {langchain_page_number}"
                )

            page = inspection_by_page_number[langchain_page_number]
            enriched_metadata = {
                **document.metadata,
                "page_number": page.page_number,
                "page_id": f"{source_name}_page_{page.page_number}",
                "text_length": page.text_length,
                "image_count": page.image_count,
                "drawing_count": page.drawing_count,
                "width": page.width,
                "height": page.height,
                "has_low_text": page.has_low_text,
            }

            document_kwargs = {
                "page_content": document.page_content,
                "metadata": enriched_metadata,
            }

            document_id = getattr(document, "id", None)
            if document_id is not None:
                document_kwargs["id"] = document_id

            enriched_documents.append(Document(**document_kwargs))

        logger.info(
            "Enriched %d document(s) with PDF inspection metadata",
            len(enriched_documents),
        )

        return enriched_documents

    except Exception:
        logger.exception("Failed to enrich documents with PDF inspection metadata")
        raise


if __name__ == "__main__":
    sample_documents = [
        Document(
            page_content="Sample annual report page text.",
            metadata={"source": "sample.pdf", "page": 0},
        )
    ]
    sample_page = PageInspection(
        page_number=1,
        text="Sample annual report page text.",
        text_length=31,
        image_count=1,
        drawing_count=2,
        width=612.0,
        height=792.0,
        has_low_text=True,
    )
    sample_inspection = PDFInspectionResult(
        source_file="sample.pdf",
        total_pages=1,
        low_text_page_count=1,
        pages=[sample_page],
    )

    enriched_sample_documents = enrich_documents_with_pdf_inspection(
        sample_documents,
        sample_inspection,
    )
    first_metadata = enriched_sample_documents[0].metadata

    logger.info(
        "Metadata enricher smoke test enriched %d document(s)",
        len(enriched_sample_documents),
    )
    logger.info(
        "First enriched document page number=%d",
        first_metadata["page_number"],
    )
    logger.info(
        "First enriched document page id=%s",
        first_metadata["page_id"],
    )
    logger.info(
        "First enriched document dimensions width=%f height=%f",
        first_metadata["width"],
        first_metadata["height"],
    )
    logger.info(
        "First enriched document has_low_text=%s",
        first_metadata["has_low_text"],
    )
