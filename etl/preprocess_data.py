import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config.config import (
    RAW_DATA_PATH, X_TRAIN_PATH, X_TEST_PATH, 
    Y_TRAIN_PATH, Y_TEST_PATH, LOGGING_CONFIG,
    REQUIRED_COLUMNS
)

# Настройка логирования
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger('data_preprocessor')

def preprocess_data():
    """
    Предобработка данных: чтение, масштабирование, разбиение на тренировочную и тестовую выборки.
    Включает проверку качества данных и обработку возможных ошибок.
    """
    try:
        logger.info("Начало предобработки данных...")
        
        # Загрузка данных
        logger.info(f"Чтение данных из {RAW_DATA_PATH}")
        df = pd.read_csv(RAW_DATA_PATH)
        
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
            # Заполнение медианными значениями
            df = df.fillna(df.median())
            logger.info("Пропущенные значения заполнены медианами")
        
        # Разделение на признаки и целевую переменную
        X = df.drop(columns=['target'])
        y = df['target']
        
        # Масштабирование признаков
        logger.info("Масштабирование признаков...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Разделение данных
        logger.info("Разделение данных на train/test...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=0.2, 
            random_state=42
        )
        
        # Сохранение обработанных данных
        pd.DataFrame(X_train).to_csv(X_TRAIN_PATH, index=False)
        pd.DataFrame(X_test).to_csv(X_TEST_PATH, index=False)
        y_train.to_csv(Y_TRAIN_PATH, index=False)
        y_test.to_csv(Y_TEST_PATH, index=False)
        
        logger.info("Предобработка успешно завершена. Данные сохранены в:")
        logger.info(f"- {X_TRAIN_PATH}")
        logger.info(f"- {X_TEST_PATH}")
        logger.info(f"- {Y_TRAIN_PATH}")
        logger.info(f"- {Y_TEST_PATH}")
        
    except Exception as e:
        logger.exception(f"Ошибка при предобработке данных: {str(e)}")
        raise