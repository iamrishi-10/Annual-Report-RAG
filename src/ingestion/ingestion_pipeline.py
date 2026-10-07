"""
Full ingestion pipeline orchestration.

This module runs the pipeline end to end: load the PDF, inspect it, enrich
metadata, chunk, clean metadata, and index the result into Pinecone. It skips
chunks already present in the index so a re-run only embeds and upserts new
or changed chunks. It does not perform retrieval, reranking, context
assembly, or generation.
"""

from src.chunking.chunker import chunk_documents
from src.document_loading.langchain_loader import load_with_langchain
from src.document_loading.pdf_loader import inspect_pdf_with_fitz
from src.preprocessing.metadata_cleaner import clean_document_metadata
from src.preprocessing.metadata_enricher import enrich_documents_with_pdf_inspection
from src.utils.logger import get_logger
from src.vectorstore.vectorstore_indexer import (
    get_existing_chunk_ids,
    index_documents_to_pinecone,
)


logger = get_logger(__name__)

PREVIEW_ID_COUNT = 5


def run_ingestion_pipeline() -> dict:
    """Run the full ingestion pipeline and index new chunks into Pinecone.

    Returns:
        A report dict summarizing the run:
        total_loaded_pages, total_chunked_documents, total_cleaned_documents,
        already_existing_documents, new_documents_indexed,
        total_documents_after_indexing, preview_existing_ids,
        preview_new_indexed_ids.
    """
    try:
        logger.info("Starting ingestion pipeline")

        documents = load_with_langchain()
        pdf_inspection = inspect_pdf_with_fitz()
        enriched_documents = enrich_documents_with_pdf_inspection(
            documents,
            pdf_inspection,
        )
        chunked_documents = chunk_documents(enriched_documents)
        cleaned_documents = clean_document_metadata(chunked_documents)

        chunk_ids = [document.metadata["chunk_id"] for document in cleaned_documents]
        existing_chunk_ids = get_existing_chunk_ids(chunk_ids)

        new_documents = [
            document
            for document in cleaned_documents
            if document.metadata["chunk_id"] not in existing_chunk_ids
        ]

        if new_documents:
            logger.info(
                "Indexing %d new chunk(s) out of %d total",
                len(new_documents),
                len(chunk_ids),
            )

            indexed_chunk_ids = index_documents_to_pinecone(new_documents)
        else:
            logger.info("All %d chunk(s) already indexed, nothing to do", len(chunk_ids))

            indexed_chunk_ids = []

        ingestion_report = {
            "total_loaded_pages": len(documents),
            "total_chunked_documents": len(chunked_documents),
            "total_cleaned_documents": len(cleaned_documents),
            "already_existing_documents": len(existing_chunk_ids),
            "new_documents_indexed": len(indexed_chunk_ids),
            "total_documents_after_indexing": len(existing_chunk_ids)
            + len(indexed_chunk_ids),
            "preview_existing_ids": sorted(existing_chunk_ids)[:PREVIEW_ID_COUNT],
            "preview_new_indexed_ids": indexed_chunk_ids[:PREVIEW_ID_COUNT],
        }

        logger.info(
            "Ingestion pipeline complete, indexed %d chunk(s)",
            len(indexed_chunk_ids),
        )

        return ingestion_report

    except Exception:
        logger.exception("Ingestion pipeline failed")
        raise
