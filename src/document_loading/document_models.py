"""
Shared document-loading data models.

These dataclasses define structured outputs for low-level PDF inspection and
hybrid document analysis stages.
"""

from dataclasses import dataclass


@dataclass
class PageInspection:
    """Inspection details for a single physical PDF page."""

    page_number: int
    text: str
    text_length: int
    image_count: int
    drawing_count: int
    width: float
    height: float
    has_low_text: bool


@dataclass
class PDFInspectionResult:
    """Inspection summary for an entire PDF document."""

    source_file: str
    total_pages: int
    low_text_page_count: int
    pages: list[PageInspection]


if __name__ == "__main__":
    sample_page = PageInspection(
        page_number=1,
        text="Sample page text.",
        text_length=17,
        image_count=0,
        drawing_count=0,
        width=612.0,
        height=792.0,
        has_low_text=True,
    )
    sample_result = PDFInspectionResult(
        source_file="sample.pdf",
        total_pages=1,
        low_text_page_count=1,
        pages=[sample_page],
    )

    print(sample_page)
    print(sample_result)
