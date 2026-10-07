"""
Verify that the Pinecone index connection works before building the upsert
module.
"""

from src.utils.logger import get_logger
from src.vectorstore.pinecone_client import get_pinecone_index


logger = get_logger(__name__)


def main() -> None:
    """Connect to Pinecone and confirm the configured index is ready."""
    index = get_pinecone_index()

    logger.info("Pinecone connection test succeeded, index is ready: %s", index)


if __name__ == "__main__":
    main()
