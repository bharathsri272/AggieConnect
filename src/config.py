"""Configuration settings for AggieConnect."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model paths
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# UC Davis URLs
UC_DAVIS_FAQ_URL = os.getenv("UC_DAVIS_FAQ_URL", "https://www.ucdavis.edu/faq")
UC_DAVIS_SERVICES_URL = os.getenv("UC_DAVIS_SERVICES_URL", "https://www.ucdavis.edu/services")
UC_DAVIS_ACADEMICS_URL = os.getenv("UC_DAVIS_ACADEMICS_URL", "https://www.ucdavis.edu/academics")

# Model Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 384))
MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", 512))

# FAISS Configuration
FAISS_INDEX_TYPE = os.getenv("FAISS_INDEX_TYPE", "IndexFlatIP")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", str(MODELS_DIR / "faiss_index.bin"))

# Data Configuration
MAX_DOCUMENTS = int(os.getenv("MAX_DOCUMENTS", 10000))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                 EMBEDDINGS_DIR, CHECKPOINTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
