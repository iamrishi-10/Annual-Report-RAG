"""
Run the full ingestion pipeline over the entire report and index every new
chunk into Pinecone.
"""

from dotenv import load_dotenv

from src.config.settings import PROCESSED_DATA_DIR, PROJECT_ROOT
from src.ingestion.ingestion_pipeline import run_ingestion_pipeline
from src.utils.json_writer import save_json
from src.utils.logger import get_logger


logger = get_logger(__name__)


def main() -> None:
    """Run the full ingestion pipeline and save the ingestion report."""
    load_dotenv(PROJECT_ROOT / ".env")

    ingestion_report = run_ingestion_pipeline()

    output_path = PROCESSED_DATA_DIR / "ingestion_report.json"

    save_json(ingestion_report, output_path)

    logger.info(
        "Ingestion pipeline indexed %d chunk(s), saved report to %s",
        ingestion_report["new_documents_indexed"],
        output_path,
    )


if __name__ == "__main__":
    main()
