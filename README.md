# Company Analysis RAG

Система для анализа компаний с использованием RAG (Retrieval-Augmented Generation) и RAG Fusion.

## Структура проекта

```
.
├── config/             # Конфигурационные файлы
│   └── settings.py     # Настройки приложения
├── models/             # Модели данных
│   └── schemas.py      # Pydantic модели
├── services/           # Бизнес-логика
│   └── rag_service.py  # RAG и RAG Fusion сервисы
├── utils/              # Вспомогательные функции
│   ├── prompts.py      # Промпты для LLM
│   └── create_collection_and_add_files_qdrant.py  # Скрипт для обработки документов
├── data/              # Директория для документов
├── main.py            # Основной скрипт
├── requirements.txt    # Зависимости проекта
├── Dockerfile         # Конфигурация Docker образа
├── docker-compose.yml # Конфигурация Docker Compose
└── .dockerignore      # Исключения для Docker сборки
```

## Установка

### Локальная установка

1. Клонируйте репозиторий
2. Установите зависимости:
```bash
pip install -r requirements.txt
```

### Установка с Docker

1. Убедитесь, что у вас установлены Docker и Docker Compose
2. Соберите и запустите контейнеры:
```bash
docker-compose up -d
```

## Настройка

Создайте файл `.env` в корневой директории проекта со следующими переменными:
```bash
OPENROUTER_API_KEY=your_api_key
QDRANT_URL=your_qdrant_url
```

## Использование

### Локальный запуск

```bash
python main.py
```

### Запуск в Docker

```bash
docker-compose up -d
```

## Особенности

- Использование RAG Fusion для первого сообщения пользователя
- Переранжирование результатов с помощью Cross Encoder
- Поддержка контекстной памяти диалога
- Логирование всех этапов работы
- Docker-поддержка для простого развертывания

## Быстрый старт

### 1. Подготовка окружения

Создайте файл `.env` в корне проекта:
```bash
OPENROUTER_API_KEY=your_api_key
QDRANT_URL=http://localhost:6333
```

### 2. Запуск с Docker Compose

```bash
docker-compose up -d
```

Это развернет:
- Qdrant базу данных на порту 6333
- Ваше приложение на порту 8000

### 3. Проверка работы Qdrant

Откройте http://localhost:6333/dashboard для доступа к веб-интерфейсу Qdrant.

Или проверьте API:
```bash
curl http://localhost:6333/collections
```

## Структура проекта

- `main.py` - основной файл приложения
- `utils/create_collection_and_add_files_qdrant.py` - обработка документов и загрузка в Qdrant
- `docker-compose.yml` - конфигурация для развертывания
- `Dockerfile` - конфигурация Docker образа
- `data/` - папка для документов
- `.env` - переменные окружения (создайте сами)

## Переменные окружения

- `OPENROUTER_API_KEY` - ключ OpenRouter (обязательно)
- `QDRANT_URL` - URL Qdrant сервера (по умолчанию: http://localhost:6333)
