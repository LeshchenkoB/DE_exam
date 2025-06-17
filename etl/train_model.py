import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import logging
from config.config import (
    X_TRAIN_PATH, Y_TRAIN_PATH, MODEL_PATH, 
    LOGGING_CONFIG, MODEL_PARAMS
)

# Настройка логирования
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger('model_trainer')

def train_model():
    """
    Обучение модели логистической регрессии на тренировочных данных.
    Включает проверку качества входных данных и обработку ошибок обучения.
    """
    try:
        logger.info("Начало обучения модели...")
        
        # Загрузка тренировочных данных
        logger.info(f"Загрузка тренировочных данных из {X_TRAIN_PATH} и {Y_TRAIN_PATH}")
        X_train = pd.read_csv(X_TRAIN_PATH)
        y_train = pd.read_csv(Y_TRAIN_PATH).values.ravel()
        
        # Проверка данных
        if X_train.empty or y_train.size == 0:
            error_msg = "Тренировочные данные отсутствуют или пусты"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        if X_train.shape[0] != y_train.shape[0]:
            error_msg = "Несоответствие размеров признаков и целевой переменной"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Создание и обучение модели
        logger.info(f"Создание модели LogisticRegression")
        model = LogisticRegression(max_iter=1000)
        
        logger.info("Обучение модели...")
        model.fit(X_train, y_train)
        
        # Сохранение модели
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
            
        logger.info(f"Модель успешно обучена и сохранена в {MODEL_PATH}")
        
    except Exception as e:
        logger.exception(f"Ошибка при обучении модели: {str(e)}")
        raise