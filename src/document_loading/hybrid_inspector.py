"""
Hybrid document-loading inspection utilities.

This module compares LangChain PDF loading output with direct PyMuPDF
inspection output. It validates that both loading paths agree on page counts
and surfaces low-text pages for later hybrid processing decisions.
"""

from pathlib import Path

from langchain_core.documents import Document

from src.document_loading.document_models import PDFInspectionResult
from src.document_loading.langchain_loader import load_with_langchain
from src.document_loading.pdf_loader import inspect_pdf_with_fitz
from src.utils.logger import get_logger


logger = get_logger(__name__)


def inspect_document_loading(
    pdf_path: Path | str | None = None,
) -> dict:
    """
    Compare LangChain loading results with direct PyMuPDF inspection results.

    Args:
        pdf_path: Optional path to a PDF file. When omitted, both loaders use
            the configured annual report path.

    Returns:
        A summary dictionary containing source file, page counts, page-count
        match status, and low-text page numbers.
    """

    try:
        logger.info("Starting hybrid document inspection")

        documents: list[Document] = load_with_langchain(pdf_path)
        pdf_inspection: PDFInspectionResult = inspect_pdf_with_fitz(pdf_path)

        langchain_page_count = len(documents)
        fitz_page_count = pdf_inspection.total_pages
        page_count_matches = langchain_page_count == fitz_page_count

        low_text_pages = [
            page.page_number
            for page in pdf_inspection.pages
            if page.has_low_text
        ]

        logger.info(
            "LangChain pages=%d | PyMuPDF pages=%d | match=%s",
            langchain_page_count,
            fitz_page_count,
            page_count_matches,
        )
        logger.info(
            "Hybrid inspection found %d low-text page(s)",
            pdf_inspection.low_text_page_count,
        )

        return {
            "source_file": pdf_inspection.source_file,
            "langchain_page_count": langchain_page_count,
            "fitz_page_count": fitz_page_count,
            "page_count_matches": page_count_matches,
            "low_text_page_count": pdf_inspection.low_text_page_count,
            "low_text_pages": low_text_pages,
        }
    except Exception:
        logger.exception("Failed during hybrid document inspection")
        raise


if __name__ == "__main__":
    inspection_summary = inspect_document_loading()

    logger.info(
        "Hybrid inspection smoke test source file is %s",
        inspection_summary["source_file"],
    )
    logger.info(
        "Hybrid inspection smoke test LangChain pages=%d PyMuPDF pages=%d",
        inspection_summary["langchain_page_count"],
        inspection_summary["fitz_page_count"],
    )
    logger.info(
        "Hybrid inspection smoke test page count match=%s",
        inspection_summary["page_count_matches"],
    )
    logger.info(
        "Hybrid inspection smoke test found %d low-text page(s)",
        inspection_summary["low_text_page_count"],
    )
