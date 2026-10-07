"""
Central configuration file for the Annual Report RAG project.

All project-wide constants should be defined here.
Business logic should not be added to this file.
"""

from pathlib import Path

# ==========================
# Project Root
# ==========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ==========================
# Directory Paths
# ==========================

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EVALUATION_DATA_DIR = DATA_DIR / "evaluation"
LOGS_DIR = PROJECT_ROOT / "logs"

# ==========================
# File Paths
# ==========================

ANNUAL_REPORT_PATH = RAW_DATA_DIR / "infosys-ar-25.pdf"
PAGES_OUTPUT_PATH = PROCESSED_DATA_DIR / "pages.json"
SKIPPED_PAGES_OUTPUT_PATH = PROCESSED_DATA_DIR / "skipped_pages.json"
CHUNKS_OUTPUT_PATH = PROCESSED_DATA_DIR / "chunks.json"

# ==========================
# Model Configuration
# ==========================

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
GENERATION_MODEL = "gpt-4o-mini"
EVALUATION_MODEL = "gpt-4o-mini"
RERANKING_MODEL = "rerank-v3.5"

# ==========================
# Pinecone Configuration
# ==========================

PINECONE_INDEX_NAME = "annual-report-rag"
PINECONE_NAMESPACE = "default"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
PINECONE_METRIC = "cosine"

# ==========================
# Chunking Configuration
# ==========================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ==========================
# Retrieval Configuration
# ==========================

TOP_K = 20          # Candidates retrieved from Pinecone
FINAL_TOP_K =   10 # Final chunks sent to the LLM after Cohere reranking
SEARCH_TYPE = "similarity"
SCORE_THRESHOLD = 0.75
FETCH_K = 40        # Pool size for MMR before diversity filtering
LAMBDA_MULT = 0.5   # MMR relevance/diversity tradeoff (1 = pure relevance)

# ==========================
# Context Configuration
# ==========================

MAX_CONTEXT_CHUNKS = 6
MAX_CONTEXT_TOKENS = 6000

# ==========================
# PDF Processing Configuration
# ==========================

MIN_PAGE_TEXT_LENGTH = 50   # Pages with less extractable text than this are flagged as image-heavy/skippable
EXTRACT_TABLES = True
TABLE_OUTPUT_FORMAT = "markdown"

# ==========================
# Evaluation Configuration
# ==========================

ENABLE_RAGAS = True
TRACK_LATENCY = True
TRACK_COST = True
TRACK_TOKEN_USAGE = True

# ==========================
# Logging Configuration
# ==========================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
