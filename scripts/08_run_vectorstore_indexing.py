"""
Run the full preprocessing pipeline and index a small chunk preview into
Pinecone.
"""

from dotenv import load_dotenv

from src.chunking.chunker import chunk_documents
from src.config.settings import PROCESSED_DATA_DIR, PROJECT_ROOT
from src.document_loading.langchain_loader import load_with_langchain
from src.document_loading.pdf_loader import inspect_pdf_with_fitz
from src.preprocessing.metadata_cleaner import clean_document_metadata
from src.preprocessing.metadata_enricher import (
    enrich_documents_with_pdf_inspection,
)
from src.utils.json_writer import save_json
from src.utils.logger import get_logger
from src.vectorstore.vectorstore_indexer import index_documents_to_pinecone


logger = get_logger(__name__)

PREVIEW_CHUNK_COUNT = 5


def main() -> None:
    """Index a small cleaned chunk preview into Pinecone."""
    load_dotenv(PROJECT_ROOT / ".env")

    documents = load_with_langchain()
    pdf_inspection = inspect_pdf_with_fitz()
    enriched_documents = enrich_documents_with_pdf_inspection(
        documents,
        pdf_inspection,
    )
    chunked_documents = chunk_documents(enriched_documents)
    cleaned_documents = clean_document_metadata(chunked_documents)
    preview_documents = cleaned_documents[:PREVIEW_CHUNK_COUNT]

    indexed_chunk_ids = index_documents_to_pinecone(preview_documents)

    output_path = PROCESSED_DATA_DIR / "vectorstore_indexing_preview.json"

    save_json(indexed_chunk_ids, output_path)

    logger.info(
        "Indexed %d chunk(s) to Pinecone and saved preview to %s",
        len(indexed_chunk_ids),
        output_path,
    )


if __name__ == "__main__":
    main()
