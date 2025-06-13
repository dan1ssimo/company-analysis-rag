from openai import OpenAI
import json

from config.settings import (
    OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL,
    DADATA_API_KEY,
    logger
)
from services.rag_service import RAGFusionClient, ConversationMemory
from services.dadata_service import DadataService
from models.schemas import ResponseFormat, COMPANY_SEARCH_TOOL
from utils.prompts import SYSTEM_PROMPT, format_context_prompt

def main():
    client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY
    )
    
    rag_client = RAGFusionClient()
    memory = ConversationMemory()
    dadata_service = DadataService(DADATA_API_KEY)

    while True:
        prompt = input("Enter your question: ")
        if prompt == "exit":
            break
        if prompt == "":
            continue
        
        logger.info(f"Получен новый запрос: {prompt}")
        
        # Сначала проверяем, нужны ли инструменты
        initial_messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        
        raw_completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=initial_messages,
            tools=[COMPANY_SEARCH_TOOL],
            tool_choice="auto"
        )
        
        raw_message = raw_completion.choices[0].message
        
        # Обработка вызова инструмента
        if raw_message.tool_calls:
            for tool_call in raw_message.tool_calls:
                if tool_call.function.name == "get_company_summary":
                    # Получаем параметры вызова
                    args = json.loads(tool_call.function.arguments)
                    inn = args.get("inn")
                    
                    # Получаем информацию о компании
                    company_info = dadata_service.get_company_summary(inn)
                    if company_info:
                        company_context = f"""
                        Информация о компании:
                        Название: {company_info.get('value', 'Н/Д')}
                        ИНН: {company_info.get('data').get('inn', 'Н/Д')}
                        КПП: {company_info.get('data').get('kpp', 'Н/Д')}
                        Руководитель: {company_info.get('data').get('management', {}).get('name', 'Н/Д')}
                        Адрес: {company_info.get('data').get('address', {}).get('value', 'Н/Д')}
                        Статус: {company_info.get('data').get('state', {}).get('status', 'Н/Д')}
                        """
                        # Добавляем оригинальное сообщение ассистента в историю
                        initial_messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": raw_message.tool_calls
                        })
                        
                        # Добавляем ответ от инструмента
                        initial_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": company_context
                        })
                        
                        # Получаем финальный ответ с учетом контекста от инструмента
                        completion = client.beta.chat.completions.parse(
                            model=OPENAI_MODEL,
                            messages=initial_messages,
                            response_format=ResponseFormat
                        )
                        message = completion.choices[0].message
                        break
        else:
            # Если инструменты не используются, применяем RAG
            is_first_message = len(memory.get_messages()) == 0
            
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
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + memory.get_messages()
            
            # Парсим ответ с учетом контекста из RAG
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
