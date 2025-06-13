from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Generator
from dataclasses import dataclass
import re
from sentence_transformers import SentenceTransformer
import PyPDF2
from PyPDF2.errors import PdfReadError, FileNotDecryptedError

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Константы
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 8000
SENTENCE_ENDINGS = r'[.!?]+\s+'
PARAGRAPH_SEPARATOR = '\n\n'

@dataclass
class ChunkMetadata:
    """Метаданные для чанка текста"""
    source: str
    filename: str
    chunk_id: int
    total_chunks: int
    chunk_size: int
    page_numbers: Optional[List[int]] = None
    section_title: Optional[str] = None

class DocumentProcessor:
    """Профессиональный процессор документов для векторной базы данных"""
    
    def __init__(self, model_name: str = 'deepvk/USER-bge-m3'):
        self.model = SentenceTransformer(model_name)
        self.qdrant_client = None
        self._initialize_qdrant_client()
    
    def _initialize_qdrant_client(self) -> None:
        """Инициализация клиента Qdrant с обработкой ошибок"""
        # Используем локальный Qdrant или переменную окружения
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        
        try:
            self.qdrant_client = QdrantClient(url=qdrant_url)
            logger.info(f"Успешно подключен к Qdrant по адресу: {qdrant_url}")
        except UnexpectedResponse as e:
            logger.error(f"Ошибка подключения к Qdrant: {e}")
            raise ConnectionError(f"Не удалось подключиться к Qdrant: {e}")

class AdvancedTextSplitter:
    """Продвинутый сплиттер текста с семантическим разделением"""
    
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = self._validate_chunk_size(chunk_size)
        self.chunk_overlap = self._validate_chunk_overlap(chunk_overlap, chunk_size)
        self.separators = separators or [
            PARAGRAPH_SEPARATOR,  # Параграфы
            '\n',                 # Переносы строк
            '. ',                 # Предложения
            '! ',                 # Восклицательные предложения
            '? ',                 # Вопросительные предложения
            ' ',                  # Пробелы
            ''                    # Символы (последний резерв)
        ]
    
    @staticmethod
    def _validate_chunk_size(chunk_size: int) -> int:
        """Валидация размера чанка"""
        if not isinstance(chunk_size, int) or chunk_size < MIN_CHUNK_SIZE:
            raise ValueError(f"Размер чанка должен быть целым числом >= {MIN_CHUNK_SIZE}")
        if chunk_size > MAX_CHUNK_SIZE:
            logger.warning(f"Большой размер чанка ({chunk_size}), рекомендуется <= {MAX_CHUNK_SIZE}")
        return chunk_size
    
    @staticmethod
    def _validate_chunk_overlap(chunk_overlap: int, chunk_size: int) -> int:
        """Валидация перекрытия чанков"""
        if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
            raise ValueError("Перекрытие чанков должно быть неотрицательным целым числом")
        if chunk_overlap >= chunk_size:
            raise ValueError("Перекрытие не может быть больше или равно размеру чанка")
        return chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """
        Рекурсивное разделение текста с использованием иерархии разделителей
        
        Args:
            text: Исходный текст для разделения
            
        Returns:
            Список чанков текста
        """
        if not text or not text.strip():
            return []
        
        # Предварительная очистка текста
        text = self._clean_text(text)
        
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        return list(self._recursive_split(text, self.separators))
    
    def _clean_text(self, text: str) -> str:
        """Очистка и нормализация текста"""
        # Удаление лишних пробелов и переносов
        text = re.sub(r'\n{3,}', '\n\n', text)  # Не более 2 переносов подряд
        text = re.sub(r' {2,}', ' ', text)      # Не более 1 пробела подряд
        text = re.sub(r'\t', ' ', text)         # Табы в пробелы
        return text.strip()
    
    def _recursive_split(self, text: str, separators: List[str]) -> Generator[str, None, None]:
        """Рекурсивное разделение текста по иерархии разделителей"""
        if not separators:
            # Если разделители закончились, принудительно разрезаем
            yield from self._force_split(text)
            return
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator == '':
            # Последний разделитель - разрезаем по символам
            yield from self._force_split(text)
            return
        
        splits = text.split(separator)
        
        current_chunk = ""
        for split in splits:
            # Восстанавливаем разделитель, кроме последнего элемента
            potential_chunk = current_chunk + (separator if current_chunk else "") + split
            
            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Текущий чанк готов
                if current_chunk.strip():
                    yield current_chunk.strip()
                
                # Если split сам по себе слишком большой, рекурсивно разделяем
                if len(split) > self.chunk_size:
                    yield from self._recursive_split(split, remaining_separators)
                    current_chunk = ""
                else:
                    current_chunk = split
        
        # Добавляем последний чанк
        if current_chunk.strip():
            yield current_chunk.strip()
    
    def _force_split(self, text: str) -> Generator[str, None, None]:
        """Принудительное разделение по размеру с учетом перекрытия"""
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk = text[start:end].strip()
            
            if chunk:
                yield chunk
            
            # Следующий чанк начинается с учетом перекрытия
            start = max(end - self.chunk_overlap, start + 1)
            
            if start >= text_length:
                break

class PDFExtractor:
    """Экстрактор текста из PDF"""
    
    @staticmethod
    def extract_text_with_metadata(pdf_path: Path) -> Tuple[str, Dict]:
        """
        Извлекает текст и метаданные из PDF файла
        
        Args:
            pdf_path: Путь к PDF файлу
            
        Returns:
            Кортеж (извлеченный_текст, метаданные)
            
        Raises:
            FileNotFoundError: Если файл не найден
            PdfReadError: Если PDF поврежден или не читается
            FileNotDecryptedError: Если PDF зашифрован
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF файл не найден: {pdf_path}")
        
        if not pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"Файл не является PDF: {pdf_path}")
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Проверка на шифрование
                if pdf_reader.is_encrypted:
                    logger.warning(f"PDF файл зашифрован: {pdf_path}")
                    raise FileNotDecryptedError(f"PDF файл зашифрован: {pdf_path}")
                
                # Извлечение метаданных
                metadata = {
                    'total_pages': len(pdf_reader.pages),
                    'file_size': pdf_path.stat().st_size,
                    'creation_time': pdf_path.stat().st_ctime,
                }
                
                # Добавление метаданных из PDF
                if pdf_reader.metadata:
                    pdf_metadata = pdf_reader.metadata
                    metadata.update({
                        'title': pdf_metadata.get('/Title', ''),
                        'author': pdf_metadata.get('/Author', ''),
                        'subject': pdf_metadata.get('/Subject', ''),
                        'creator': pdf_metadata.get('/Creator', ''),
                    })
                
                # Извлечение текста
                text_parts = []
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_parts.append(f"[Страница {page_num}]\n{page_text}")
                        else:
                            logger.warning(f"Пустая страница {page_num} в файле {pdf_path}")
                    except Exception as page_error:
                        logger.error(f"Ошибка извлечения текста со страницы {page_num}: {page_error}")
                        continue
                
                full_text = '\n\n'.join(text_parts)
                
                if not full_text.strip():
                    raise ValueError(f"Не удалось извлечь текст из PDF: {pdf_path}")
                
                logger.info(f"Успешно извлечен текст из {pdf_path} ({len(full_text)} символов)")
                return full_text, metadata
                
        except (PdfReadError, FileNotDecryptedError):
            raise
        except PermissionError:
            raise PermissionError(f"Нет прав доступа к файлу: {pdf_path}")
        except Exception as e:
            logger.error(f"Неожиданная ошибка при чтении PDF {pdf_path}: {e}")
            raise RuntimeError(f"Ошибка обработки PDF файла: {e}")


def add_documents_to_qdrant(collection_name: str, documents: List[str], metadata: Optional[List[Dict]] = None) -> str:
    """Добавляет документы в коллекцию Qdrant с улучшенной обработкой ошибок"""
    if not collection_name:
        raise ValueError("Название коллекции не может быть пустым")
    
    if not documents:
        raise ValueError("Список документов не может быть пустым")
    
    try:
        processor = DocumentProcessor()
        
        # Получение эмбеддингов
        embeddings = []
        for i, doc in enumerate(documents):
            if not doc.strip():
                logger.warning(f"Пустой документ по индексу {i}, пропускаем")
                continue
            
            try:
                embedding = processor.model.encode(doc, normalize_embeddings=True)
                embeddings.append(embedding)
            except Exception as embed_error:
                logger.error(f"Ошибка создания эмбеддинга для документа {i}: {embed_error}")
                raise RuntimeError(f"Ошибка создания эмбеддинга: {embed_error}")
        
        # Подготовка точек
        points = []
        for i, (embedding, doc) in enumerate(zip(embeddings, documents)):
            point = models.PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={"text": doc, "doc_id": i}
            )
            
            if metadata and i < len(metadata):
                point.payload.update(metadata[i])
            
            points.append(point)
        
        # Загрузка в Qdrant
        processor.qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        logger.info(f"Успешно добавлено {len(points)} документов в коллекцию {collection_name}")
        return f"Успешно добавлено {len(points)} документов в коллекцию {collection_name}"
        
    except (ValueError, RuntimeError):
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка при добавлении документов: {e}")
        raise RuntimeError(f"Ошибка добавления документов в Qdrant: {e}")

def add_pdf_documents_to_qdrant(
    collection_name: str, 
    pdf_paths: List[str], 
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> str:
    """Профессиональное добавление PDF документов в Qdrant"""
    if not collection_name:
        raise ValueError("Название коллекции не может быть пустым")
    
    if not pdf_paths:
        raise ValueError("Список PDF файлов не может быть пустым")
    
    # Валидация путей
    valid_paths = []
    for path_str in pdf_paths:
        path = Path(path_str)
        if path.exists() and path.suffix.lower() == '.pdf':
            valid_paths.append(path)
        else:
            logger.warning(f"Файл не найден или не является PDF: {path_str}")
    
    if not valid_paths:
        raise FileNotFoundError("Не найдено валидных PDF файлов")
    
    # Инициализация компонентов
    extractor = PDFExtractor()
    splitter = AdvancedTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    all_documents = []
    all_metadata = []
    
    for pdf_path in valid_paths:
        try:
            logger.info(f"Обрабатываю файл: {pdf_path}")
            
            # Извлечение текста и метаданных
            text, pdf_metadata = extractor.extract_text_with_metadata(pdf_path)
            
            # Разбивка на чанки
            chunks = splitter.split_text(text)
            
            if not chunks:
                logger.warning(f"Не удалось создать чанки для файла: {pdf_path}")
                continue
            
            # Создание метаданных для каждого чанка
            for i, chunk in enumerate(chunks):
                chunk_metadata = ChunkMetadata(
                    source=str(pdf_path),
                    filename=pdf_path.name,
                    chunk_id=i,
                    total_chunks=len(chunks),
                    chunk_size=len(chunk)
                )
                
                # Объединение с метаданными PDF
                metadata_dict = {
                    **pdf_metadata,
                    'source': chunk_metadata.source,
                    'filename': chunk_metadata.filename,
                    'chunk_id': chunk_metadata.chunk_id,
                    'total_chunks': chunk_metadata.total_chunks,
                    'chunk_size': chunk_metadata.chunk_size,
                }
                
                all_documents.append(chunk)
                all_metadata.append(metadata_dict)
            
            logger.info(f"Создано {len(chunks)} чанков для файла {pdf_path.name}")
            
        except (FileNotFoundError, PdfReadError, FileNotDecryptedError, ValueError) as e:
            logger.error(f"Ошибка обработки файла {pdf_path}: {e}")
            continue
        except Exception as e:
            logger.error(f"Неожиданная ошибка при обработке {pdf_path}: {e}")
            continue
    
    if not all_documents:
        raise RuntimeError("Не удалось обработать ни одного PDF файла")
    
    return add_documents_to_qdrant(collection_name, all_documents, all_metadata)

def create_collection(
    collection_name: str,
    vector_size: Optional[int] = None,
    distance_metric: str = "Cosine",
    recreate_if_exists: bool = False
) -> str:
    """
    Создает новую коллекцию в Qdrant
    
    Args:
        collection_name: Название коллекции
        vector_size: Размер векторов (если не указан, определяется автоматически)
        distance_metric: Метрика расстояния (Cosine, Dot, Euclid)
        recreate_if_exists: Пересоздать коллекцию, если она уже существует
        
    Returns:
        Сообщение о результате создания коллекции
        
    Raises:
        ValueError: При неверных параметрах
        ConnectionError: При проблемах с подключением к Qdrant
        RuntimeError: При других ошибках
    """
    if not collection_name or not collection_name.strip():
        raise ValueError("Название коллекции не может быть пустым")
    
    # Валидация метрики расстояния
    valid_distances = ["Cosine", "Dot", "Euclid"]
    if distance_metric not in valid_distances:
        raise ValueError(f"Неподдерживаемая метрика расстояния. Доступные: {valid_distances}")
    
    try:
        processor = DocumentProcessor()
        
        # Определение размера векторов, если не указан
        if vector_size is None:
            logger.info("Определяю размер векторов для модели...")
            test_text = "Тестовый текст для определения размера эмбеддинга"
            test_embedding = processor.model.encode(test_text, normalize_embeddings=True)
            vector_size = len(test_embedding)
            logger.info(f"Размер векторов определен автоматически: {vector_size}")
        
        # Проверка существования коллекции
        try:
            collections = processor.qdrant_client.get_collections()
            existing_collections = [col.name for col in collections.collections]
            
            if collection_name in existing_collections:
                if recreate_if_exists:
                    logger.info(f"Удаляю существующую коллекцию: {collection_name}")
                    processor.qdrant_client.delete_collection(collection_name)
                else:
                    message = f"Коллекция '{collection_name}' уже существует"
                    logger.warning(message)
                    return message
        except Exception as e:
            logger.warning(f"Не удалось проверить существующие коллекции: {e}")
        
        # Создание коллекции
        processor.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=getattr(models.Distance, distance_metric.upper())
            )
        )
        
        success_message = f"Коллекция '{collection_name}' успешно создана с размером векторов {vector_size} и метрикой {distance_metric}"
        logger.info(success_message)
        return success_message
        
    except ValueError:
        raise
    except UnexpectedResponse as e:
        error_message = f"Ошибка подключения к Qdrant при создании коллекции: {e}"
        logger.error(error_message)
        raise ConnectionError(error_message)
    except Exception as e:
        error_message = f"Неожиданная ошибка при создании коллекции '{collection_name}': {e}"
        logger.error(error_message)
        raise RuntimeError(error_message)


def get_collection_info(collection_name: str) -> Dict:
    """
    Получает информацию о коллекции
    
    Args:
        collection_name: Название коллекции
        
    Returns:
        Словарь с информацией о коллекции
        
    Raises:
        ValueError: При неверных параметрах
        ConnectionError: При проблемах с подключением к Qdrant
        RuntimeError: При других ошибках
    """
    if not collection_name or not collection_name.strip():
        raise ValueError("Название коллекции не может быть пустым")
    
    try:
        processor = DocumentProcessor()
        
        # Получение информации о коллекции
        collection_info = processor.qdrant_client.get_collection(collection_name)
        
        # Получение статистики
        collection_stats = processor.qdrant_client.count(collection_name)
        
        info = {
            "name": collection_name,
            "vectors_count": collection_stats.count,
            "vector_size": collection_info.config.params.vectors.size,
            "distance_metric": collection_info.config.params.vectors.distance.name,
            "status": collection_info.status,
            "optimizer_status": collection_info.optimizer_status,
            "indexed_vectors_count": collection_info.indexed_vectors_count if hasattr(collection_info, 'indexed_vectors_count') else None
        }
        
        logger.info(f"Получена информация о коллекции '{collection_name}'")
        return info
        
    except ValueError:
        raise
    except UnexpectedResponse as e:
        error_message = f"Коллекция '{collection_name}' не найдена или ошибка подключения: {e}"
        logger.error(error_message)
        raise ConnectionError(error_message)
    except Exception as e:
        error_message = f"Ошибка получения информации о коллекции '{collection_name}': {e}"
        logger.error(error_message)
        raise RuntimeError(error_message)


def list_collections() -> List[str]:
    """
    Возвращает список всех коллекций
    
    Returns:
        Список названий коллекций
        
    Raises:
        ConnectionError: При проблемах с подключением к Qdrant
        RuntimeError: При других ошибках
    """
    try:
        processor = DocumentProcessor()
        collections = processor.qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        logger.info(f"Найдено коллекций: {len(collection_names)}")
        return collection_names
        
    except UnexpectedResponse as e:
        error_message = f"Ошибка подключения к Qdrant при получении списка коллекций: {e}"
        logger.error(error_message)
        raise ConnectionError(error_message)
    except Exception as e:
        error_message = f"Ошибка получения списка коллекций: {e}"
        logger.error(error_message)
        raise RuntimeError(error_message)


if __name__ == "__main__":
    try:
        # Создание коллекции перед добавлением документов
        logger.info("Создаю коллекцию...")
        collection_result = create_collection(
            collection_name="my_documents",
            distance_metric="Cosine",
            recreate_if_exists=True
        )
        print(collection_result)
        
        # Получение информации о коллекции
        logger.info("Получаю информацию о коллекции...")
        collection_info = get_collection_info("my_documents")
        print(f"Информация о коллекции: {collection_info}")
        
        # Добавление конкретных PDF файлов
        result = add_pdf_documents_to_qdrant(
            collection_name="my_documents",
            pdf_paths=["data/alpha_company.pdf", "data/beta_company.pdf", "data/gamma_company.pdf"],
            chunk_size=1200,
            chunk_overlap=200
        )
        
        # Получение обновленной информации о коллекции
        logger.info("Получаю обновленную информацию о коллекции...")
        updated_info = get_collection_info("my_documents")
        print(f"Обновленная информация о коллекции: {updated_info}")
        
    except Exception as e:
        logger.error(f"Ошибка выполнения: {e}")
        raise