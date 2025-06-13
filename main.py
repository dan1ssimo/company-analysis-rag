from openai import OpenAI
import os
from config.settings import (
    OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL,
    logger
)
from services.rag_service import RAGFusionClient, ConversationMemory
from models.schemas import ResponseFormat
from utils.prompts import SYSTEM_PROMPT, format_context_prompt

def main():
    client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY
    )
    
    rag_client = RAGFusionClient()
    memory = ConversationMemory()

    while True:
        prompt = input("Enter your question: ")
        if prompt == "exit":
            break
        if prompt == "":
            continue
        
        logger.info(f"Получен новый запрос: {prompt}")
        
        # Проверяем, является ли это первым сообщением
        is_first_message = len(memory.get_messages()) == 0
        
        # Выбираем метод поиска в зависимости от того, первое ли это сообщение
        if is_first_message:
            logger.info("Применяем RAG Fusion для первого сообщения")
            relevant_docs = rag_client.search_with_fusion(prompt)
        else:
            logger.info("Используем стандартный поиск RAG")
            relevant_docs = rag_client.search_relevant_documents(prompt)
            
        context = "\n\n".join([f"Source: {doc['source']}\nText: {doc['text']}" for doc in relevant_docs])
        
        # Формируем промпт с контекстом
        user_prompt = format_context_prompt(context, prompt)
        
        # Добавляем сообщение пользователя в память
        memory.add_message("user", user_prompt)
        
        # Формируем список сообщений для API
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + memory.get_messages()
        
        completion = client.beta.chat.completions.parse(
            model=OPENAI_MODEL,
            messages=messages,
            response_format=ResponseFormat
        )
        message = completion.choices[0].message
        if message.parsed:
            # Добавляем ответ ассистента в память
            memory.add_message("assistant", message.parsed.final_answer)
            
            logger.info("Шаги рассуждения:")
            for step in message.parsed.steps:
                logger.info(f"Шаг: {step.step}")
                logger.info(f"Рассуждение: {step.reasoning}")
                logger.info(f"Источники: {', '.join(step.sources)}")
            logger.info(f"Итоговый ответ: {message.parsed.final_answer}")
        else:
            logger.error(f"Ошибка: {message.refusal}")

if __name__ == "__main__":
    main()
