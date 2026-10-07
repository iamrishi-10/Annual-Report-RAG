"""
LangChain PDF loading utilities.

This module is responsible only for loading a PDF through LangChain's
PyMuPDFLoader and returning one LangChain Document per PDF page.
"""

from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

from src.config.settings import ANNUAL_REPORT_PATH
from src.utils.logger import get_logger


logger = get_logger(__name__)


def load_with_langchain(
    pdf_path: Path | str | None = None,
) -> list[Document]:
    """
    Load a PDF through LangChain and return one Document per PDF page.

    Args:
        pdf_path: Optional path to a PDF file. When omitted, the configured
            annual report path is used.

    Returns:
        A list of LangChain Document objects, one per loaded PDF page.

    Raises:
        FileNotFoundError: If the resolved PDF path does not exist.
        ValueError: If the resolved path is not a PDF file, or if no pages are
            loaded from the PDF.

    Underlying loader errors are logged with traceback and propagated.
    """

    resolved_path: Path | None = None

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

        logger.info("Loading PDF with LangChain from %s", resolved_path)

        loader = PyMuPDFLoader(str(resolved_path))
        documents = loader.load()

        if not documents:
            raise ValueError(f"No pages were loaded from PDF: {resolved_path}")

        logger.info(
            "Loaded %d document/page(s) from %s",
            len(documents),
            resolved_path,
        )
        return documents
    except Exception:
        failed_path = resolved_path if resolved_path is not None else pdf_path
        logger.exception("Failed to load PDF with LangChain from %s", failed_path)
        raise


if __name__ == "__main__":
    loaded_documents = load_with_langchain()
    logger.info(
        "LangChain loader smoke test loaded %d document/page(s)",
        len(loaded_documents),
    )
    logger.info(
        "LangChain loader first document type is %s",
        type(loaded_documents[0]).__name__,
    )
