import pandas as pd
from sklearn.datasets import load_breast_cancer
import logging
from config.config import RAW_DATA_PATH, LOGGING_CONFIG, REQUIRED_COLUMNS

# Настройка логирования
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger('data_loader')

def load_and_save_data() -> None:
    """
    Загрузка набора данных о раке груди и сохранение в CSV-файл.
    Обрабатывает возможные ошибки загрузки данных и проверяет структуру данных.
    """
    try:
        logger.info("Начало загрузки breast cancer dataset...")
        
        # Загрузка данных из sklearn
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        
        # Проверка структуры данных
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            error_msg = f"Отсутствуют обязательные колонки: {', '.join(missing_columns)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Проверка на пропущенные значения
        if df.isnull().any().any():
            null_counts = df.isnull().sum()
            logger.warning(f"Обнаружены пропущенные значения:\n{null_counts[null_counts > 0]}")
        
        # Сохранение данных
        df.to_csv(RAW_DATA_PATH, index=False)
        logger.info(f"Данные успешно сохранены в {RAW_DATA_PATH}")
        
    except Exception as e:
        logger.exception(f"Критическая ошибка при загрузке данных: {str(e)}")
        raise