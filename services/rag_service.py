from typing import List, Dict, Deque
from collections import deque
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from config.settings import (
    OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL,
    QDRANT_URL, COLLECTION_NAME, EMBEDDING_MODEL,
    CROSS_ENCODER_MODEL, MAX_LENGTH, MAX_MEMORY_MESSAGES,
    logger
)

class ConversationMemory:
    def __init__(self, max_messages: int = MAX_MEMORY_MESSAGES):
        self.max_messages = max_messages
        self.messages: Deque[Dict] = deque(maxlen=max_messages)
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
    
    def get_messages(self) -> List[Dict]:
        return list(self.messages)
    
    def clear(self):
        self.messages.clear()

class RAGClient:
    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.qdrant_client = QdrantClient(url=QDRANT_URL)
        self.collection_name = collection_name

    def search_relevant_documents(self, query: str, limit: int = 3) -> List[Dict]:
        query_vector = self.model.encode([query])[0].tolist()
        search_result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True
        )
        return [
            {
                "text": hit.payload["text"],
                "source": hit.payload["source"],
                "score": hit.score
            }
            for hit in search_result.points
        ]

class RAGFusionClient(RAGClient):
    def __init__(self, collection_name: str = COLLECTION_NAME):
        super().__init__(collection_name)
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, max_length=MAX_LENGTH)
        
    def generate_query_variations(self, query: str, num_variations: int = 3) -> List[str]:
        """Генерирует вариации исходного запроса с помощью LLM."""
        system_prompt = """Ты - помощник по переформулировке запросов. 
        Сгенерируй несколько разных способов задать тот же вопрос, используя другие слова и формулировки.
        Каждая переформулировка должна сохранять основной смысл, но использовать разные слова и структуру предложения.
        Не нумеруй переформулировки."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Переформулируй следующий запрос {num_variations} раз:\n{query}"}
        ]
        
        client = OpenAI(
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY
        )
        
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages
        )
        
        variations = [query]  # Добавляем оригинальный запрос
        variations.extend([line.strip() for line in completion.choices[0].message.content.split('\n') if line.strip()])
        variations = variations[:num_variations + 1]  # +1 потому что включаем оригинальный запрос
        
        logger.info("Сгенерированные вариации запроса:")
        for i, variation in enumerate(variations, 1):
            logger.info(f"{i}. {variation}")
            
        return variations
    
    def rerank_documents(self, query: str, documents: List[Dict]) -> List[Dict]:
        """Переранжирует документы с помощью Cross Encoder."""
        if not documents:
            return []
            
        # Подготавливаем пары запрос-документ для Cross Encoder
        pairs = [(query, doc["text"]) for doc in documents]
        
        # Получаем скоры от Cross Encoder
        scores = self.cross_encoder.predict(pairs)
        
        # Объединяем документы со скоррами и сортируем
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored_docs]
    
    def search_with_fusion(self, query: str, limit: int = 3) -> List[Dict]:
        """Выполняет поиск с использованием RAG Fusion."""
        # Генерируем вариации запроса
        query_variations = self.generate_query_variations(query)
        
        # Собираем документы для каждой вариации
        all_documents = []
        for variation in query_variations:
            docs = self.search_relevant_documents(variation, limit=limit)
            all_documents.extend(docs)
        
        # Удаляем дубликаты по тексту
        unique_docs = {doc["text"]: doc for doc in all_documents}.values()
        
        # Переранжируем документы
        reranked_docs = self.rerank_documents(query, list(unique_docs))
        
        return reranked_docs[:limit] 