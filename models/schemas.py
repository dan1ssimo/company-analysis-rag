from pydantic import BaseModel, Field
from typing import List

class Step(BaseModel):
    step: str = Field(description="Шаг в процессе рассуждения")
    reasoning: str = Field(description="Рассуждение на шаге")
    sources: list[str] = Field(description="Источники, на которых основывается шаг")

class ResponseFormat(BaseModel):
    steps: list[Step] = Field(description="Шаги в процессе рассуждения")
    final_answer: str = Field(description="Итоговый ответ на вопрос пользователя")

COMPANY_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "get_company_summary",
        "description": "Получает информацию о компании по ИНН. Вызывай этот инструмент если пользователь запросил краткую сводку или общую информацию о компании по ИНН.",
        "parameters": {
            "type": "object",
            "properties": {
                "inn": {
                    "type": "string",
                    "description": "ИНН компании (10 или 12 цифр)"
                }
            },
            "required": ["inn"],
            "additionalProperties": False
        },
        "strict": True
    }
} 