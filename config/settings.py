import os
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# OpenAI settings
OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"

# Qdrant settings
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "my_documents"

# RAG settings
EMBEDDING_MODEL = 'deepvk/USER-bge-m3'
CROSS_ENCODER_MODEL = "PitKoro/cross-encoder-ru-msmarco-passage"
MAX_LENGTH = 512
MAX_MEMORY_MESSAGES = 5 