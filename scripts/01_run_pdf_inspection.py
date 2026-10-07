"""
Run direct PyMuPDF PDF inspection and save the result as JSON.
"""

from src.config.settings import PROCESSED_DATA_DIR
from src.document_loading.pdf_loader import inspect_pdf_with_fitz
from src.utils.json_writer import save_json
from src.utils.logger import get_logger
from src.utils.serialization import to_json_safe


logger = get_logger(__name__)


def main() -> None:
    """Inspect the configured annual report PDF and save the result as JSON."""
    inspection_result = inspect_pdf_with_fitz()
    json_safe_result = to_json_safe(inspection_result)
    output_path = PROCESSED_DATA_DIR / "pdf_inspection.json"

    save_json(json_safe_result, output_path)

    logger.info("PDF inspection completed and saved to %s", output_path)


if __name__ == "__main__":
    main()
