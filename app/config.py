import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "4"))
# If even the best chunk is below this cosine similarity, we skip the LLM
# and answer "not available" directly (cheaper and avoids hallucination).
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.25"))

DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
MAX_UPLOAD_MB = 5

NOT_FOUND_MESSAGE = "The requested information is not available in the document."
