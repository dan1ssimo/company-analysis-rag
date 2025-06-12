from openai import OpenAI
from langgraph.graph import StateGraph, START, END
import os
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import List, Dict


class Step(BaseModel):
    step: str = Field(description="The step of the reasoning process")
    reasoning: str = Field(description="The reasoning process of the step")
    sources: list[str] = Field(description="The sources of the step")

class ResponseFormat(BaseModel):
    steps: list[Step] = Field(description="The steps of the reasoning process")
    final_answer: str = Field(description="The final answer to the user's question")


class RAGClient:
    def __init__(self, collection_name: str = "my_documents"):
        self.model = SentenceTransformer('deepvk/USER-bge-m3')
        self.qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
        self.collection_name = collection_name

    def search_relevant_documents(self, query: str, limit: int = 3) -> List[Dict]:
        query_vector = self.model.encode(query).tolist()
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        return [
            {
                "text": hit.payload.get("text", ""),
                "source": hit.payload.get("source", ""),
                "score": hit.score
            }
            for hit in search_result
        ]


if __name__ == "__main__":
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    
    rag_client = RAGClient()

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
        
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=ResponseFormat
        )
        message = completion.choices[0].message
        if message.parsed:
            print("\nSteps:")
            for step in message.parsed.steps:
                print(f"\nStep: {step.step}")
                print(f"Reasoning: {step.reasoning}")
                print(f"Sources: {', '.join(step.sources)}")
            print("\nFinal answer:", message.parsed.final_answer)
        else:
            print(message.refusal)
