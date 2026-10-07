"""
Run hybrid document-loading inspection and save the summary as JSON.
"""

from src.config.settings import PROCESSED_DATA_DIR
from src.document_loading.hybrid_inspector import inspect_document_loading
from src.utils.json_writer import save_json
from src.utils.logger import get_logger


logger = get_logger(__name__)


def main() -> None:
    """Inspect both loading paths and save the hybrid summary as JSON."""
    inspection_summary = inspect_document_loading()
    output_path = PROCESSED_DATA_DIR / "hybrid_inspection.json"

    save_json(inspection_summary, output_path)

    logger.info("Hybrid inspection completed and saved to %s", output_path)


if __name__ == "__main__":
    main()
