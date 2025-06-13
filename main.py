from openai import OpenAI
from langgraph.graph import StateGraph, START, END
import os
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Deque
from collections import deque


class Step(BaseModel):
    step: str = Field(description="Шаг в процессе рассуждения")
    reasoning: str = Field(description="Рассуждение на шаге")
    sources: list[str] = Field(description="Источники, на которых основывается шаг")

class ResponseFormat(BaseModel):
    steps: list[Step] = Field(description="Шаги в процессе рассуждения")
    final_answer: str = Field(description="Итоговый ответ на вопрос пользователя")


class ConversationMemory:
    def __init__(self, max_messages: int = 5):
        self.max_messages = max_messages
        self.messages: Deque[Dict] = deque(maxlen=max_messages)
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
    
    def get_messages(self) -> List[Dict]:
        return list(self.messages)
    
    def clear(self):
        self.messages.clear()


class RAGClient:
    def __init__(self, collection_name: str = "my_documents"):
        self.model = SentenceTransformer('deepvk/USER-bge-m3')
        self.qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
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


if __name__ == "__main__":
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    
    rag_client = RAGClient()
    memory = ConversationMemory()

    while True:
        prompt = input("Enter your question: ")
        if prompt == "exit":
            break
        if prompt == "":
            continue
        
        # Получаем релевантные документы
        relevant_docs = rag_client.search_relevant_documents(prompt)
        context = "\n\n".join([f"Source: {doc['source']}\nText: {doc['text']}" for doc in relevant_docs])
        
        # Формируем промпт с контекстом
        system_prompt = """Ты — полезный ассистент. Используй предоставленный контекст, чтобы ответить на вопрос пользователя.
        Если в контексте недостаточно информации для ответа, скажи об этом и уточни детали.
        Всегда указывай источники, когда предоставляешь информацию."""
        
        user_prompt = f"""
        Контекст:
        {context}
        
        Вопрос: {prompt}
        """
        
        # Добавляем сообщение пользователя в память
        memory.add_message("user", user_prompt)
        
        # Формируем список сообщений для API
        messages = [{"role": "system", "content": system_prompt}] + memory.get_messages()
        
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format=ResponseFormat
        )
        message = completion.choices[0].message
        if message.parsed:
            # Добавляем ответ ассистента в память
            memory.add_message("assistant", message.parsed.final_answer)
            
            print("\nSteps:")
            for step in message.parsed.steps:
                print(f"\nStep: {step.step}")
                print(f"Reasoning: {step.reasoning}")
                print(f"Sources: {', '.join(step.sources)}")
            print("\nFinal answer:", message.parsed.final_answer)
        else:
            print(message.refusal)
