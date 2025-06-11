# Company Analysis RAG

Система анализа компаний с использованием RAG (Retrieval-Augmented Generation)

## Быстрый старт

### 1. Подготовка окружения

Создайте файл `.env` в корне проекта:
```bash

OPENAI_API_KEY=your_openai_api_key_here
QDRANT_URL=http://localhost:6333
```

### 2. Запуск с Docker Compose

```bash
docker-compose up -d
```

Это развернет:
- Qdrant базу данных на порту 6333
- Ваше приложение на порту 8000 (при полном запуске)

### 3. Проверка работы Qdrant

Откройте http://localhost:6333/dashboard для доступа к веб-интерфейсу Qdrant.

Или проверьте API:
```bash
curl http://localhost:6333/collections
```

## Структура проекта

- `main.py` - основной файл приложения
- `create_collection_and_add_files_qdrant.py` - обработка документов и загрузка в Qdrant
- `docker-compose.yml` - конфигурация для развертывания
- `data/` - папка для документов
- `.env` - переменные окружения (создайте сами)

## Переменные окружения

- `OPENAI_API_KEY` - ключ OpenAI API (обязательно)
- `QDRANT_URL` - URL Qdrant сервера (по умолчанию: http://localhost:6333)
