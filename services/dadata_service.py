from dadata import Dadata
from config.settings import logger

class DadataService:
    def __init__(self, api_key: str):
        self.client = Dadata(api_key)
    
    def get_company_summary(self, inn: str) -> dict:
        """
        Получает информацию о компании по ИНН
        
        Args:
            inn (str): ИНН компании
            
        Returns:
            dict: Информация о компании
        """
        try:
            result = self.client.find_by_id("party", inn)
            if result:
                return result[0]  # Возвращаем первую найденную компанию
            return None
        except Exception as e:
            logger.error(f"Ошибка при получении данных из DaData: {str(e)}")
            return None 