"""
Pinecone client and index setup utilities.

This module handles Pinecone client initialization and ensures the configured
index exists, creating it if necessary. It does not embed documents, upsert
vectors, or perform retrieval.
"""

import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pinecone.db_data.index import Index

from src.config.api_keys import is_key_set
from src.config.settings import (
    EMBEDDING_DIMENSION,
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_METRIC,
    PINECONE_REGION,
    PROJECT_ROOT,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)

load_dotenv(PROJECT_ROOT / ".env")


def get_pinecone_client() -> Pinecone:
    """Create and return a Pinecone client.

    Returns:
        A configured Pinecone client.

    Raises:
        ValueError: If PINECONE_API_KEY is not set.
    """
    try:
        if not is_key_set("PINECONE_API_KEY"):
            raise ValueError("PINECONE_API_KEY must be set.")

        api_key = os.getenv("PINECONE_API_KEY")

        logger.info("Initializing Pinecone client")

        client = Pinecone(api_key=api_key)

        logger.info("Pinecone client initialized")

        return client

    except Exception:
        logger.exception("Failed to initialize Pinecone client")
        raise


def get_pinecone_index() -> Index:
    """Return a ready-to-use Pinecone index, creating it if missing.

    Returns:
        The Pinecone index configured with EMBEDDING_DIMENSION and cosine
        metric.

    Raises:
        ValueError: If PINECONE_INDEX_NAME is empty, EMBEDDING_DIMENSION is
            not a positive integer, or PINECONE_CLOUD, PINECONE_REGION, or
            PINECONE_METRIC is empty.
    """
    try:
        if not PINECONE_INDEX_NAME.strip():
            raise ValueError("PINECONE_INDEX_NAME must not be empty.")

        if EMBEDDING_DIMENSION <= 0:
            raise ValueError("EMBEDDING_DIMENSION must be greater than 0.")

        if not PINECONE_CLOUD.strip():
            raise ValueError("PINECONE_CLOUD must not be empty.")

        if not PINECONE_REGION.strip():
            raise ValueError("PINECONE_REGION must not be empty.")

        if not PINECONE_METRIC.strip():
            raise ValueError("PINECONE_METRIC must not be empty.")

        client = get_pinecone_client()

        logger.info("Checking Pinecone index %s", PINECONE_INDEX_NAME)

        if client.has_index(PINECONE_INDEX_NAME):
            logger.info("Using existing Pinecone index %s", PINECONE_INDEX_NAME)
        else:
            logger.info("Creating Pinecone index %s", PINECONE_INDEX_NAME)

            client.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric=PINECONE_METRIC,
                spec=ServerlessSpec(
                    cloud=PINECONE_CLOUD,
                    region=PINECONE_REGION,
                ),
            )

        index = client.Index(PINECONE_INDEX_NAME)

        logger.info("Pinecone index %s ready", PINECONE_INDEX_NAME)

        return index

    except Exception:
        logger.exception("Failed to get Pinecone index %s", PINECONE_INDEX_NAME)
        raise
