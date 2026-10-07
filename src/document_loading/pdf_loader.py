"""
Low-level PDF inspection utilities using PyMuPDF.

This module inspects physical PDF pages directly using fitz and collects
page-level signals such as text length, image count, drawing count, and page
dimensions. It does not create LangChain Document objects.
"""

from pathlib import Path

import fitz

from src.config.settings import ANNUAL_REPORT_PATH, MIN_PAGE_TEXT_LENGTH
from src.document_loading.document_models import PageInspection, PDFInspectionResult
from src.utils.logger import get_logger


logger = get_logger(__name__)


def inspect_pdf_with_fitz(
    pdf_path: Path | str | None = None,
) -> PDFInspectionResult:
    """
    Inspect PDF pages directly with PyMuPDF and return page-level signals.

    Args:
        pdf_path: Optional path to a PDF file. When omitted, the configured
            annual report path is used.

    Returns:
        A PDFInspectionResult containing the source file, total page count,
        low-text page count, and page-level inspection details.

    Raises:
        FileNotFoundError: If the resolved PDF path does not exist.
        ValueError: If the resolved path is not a PDF file, or if the PDF has
            no pages.
    """

    resolved_path: Path | None = None
    pdf_document = None

    try:
        resolved_path = (
            Path(pdf_path)
            if pdf_path is not None
            else ANNUAL_REPORT_PATH
        )

        if not resolved_path.exists():
            raise FileNotFoundError(f"PDF path does not exist: {resolved_path}")

        if not resolved_path.is_file():
            raise ValueError(f"PDF path is not a file: {resolved_path}")

        if resolved_path.suffix.lower() != ".pdf":
            raise ValueError(f"PDF path must have a .pdf extension: {resolved_path}")

        logger.info("Inspecting PDF with PyMuPDF from %s", resolved_path)

        pdf_document = fitz.open(resolved_path)

        if pdf_document.page_count == 0:
            raise ValueError(f"No pages found in PDF: {resolved_path}")

        page_inspections: list[PageInspection] = []

        for page_index in range(pdf_document.page_count):
            page = pdf_document[page_index]
            text = page.get_text()
            stripped_text_length = len(text.strip())

            page_inspections.append(
                PageInspection(
                    page_number=page_index + 1,
                    text=text,
                    text_length=stripped_text_length,
                    image_count=len(page.get_images(full=True)),
                    drawing_count=len(page.get_drawings()),
                    width=page.rect.width,
                    height=page.rect.height,
                    has_low_text=stripped_text_length < MIN_PAGE_TEXT_LENGTH,
                )
            )

        low_text_page_count = sum(
            1 for page in page_inspections if page.has_low_text
        )

        logger.info(
            "Inspected %d page(s) from %s",
            len(page_inspections),
            resolved_path,
        )
        logger.info(
            "Found %d low-text page(s) in %s",
            low_text_page_count,
            resolved_path,
        )

        return PDFInspectionResult(
            source_file=str(resolved_path),
            total_pages=pdf_document.page_count,
            low_text_page_count=low_text_page_count,
            pages=page_inspections,
        )
    except Exception:
        failed_path = resolved_path if resolved_path is not None else pdf_path
        logger.exception("Failed to inspect PDF with PyMuPDF from %s", failed_path)
        raise
    finally:
        if pdf_document is not None:
            pdf_document.close()


if __name__ == "__main__":
    inspection_result = inspect_pdf_with_fitz()
    first_page = inspection_result.pages[0]

    logger.info(
        "PyMuPDF inspection smoke test inspected %d page(s)",
        inspection_result.total_pages,
    )
    logger.info(
        "PyMuPDF inspection smoke test found %d low-text page(s)",
        inspection_result.low_text_page_count,
    )
    logger.info(
        "First page dimensions are width=%f height=%f",
        first_page.width,
        first_page.height,
    )
    logger.info(
        "Type of the first page inspection result is %s",
        type(first_page),
    )
