import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import json
import logging
import numpy as np
from config.config import (
    X_TEST_PATH, Y_TEST_PATH, MODEL_PATH, 
    METRICS_PATH, LOGGING_CONFIG
)

# Настройка логирования
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger('model_evaluator')

def evaluate_model():
    """
    Оценка модели на тестовых данных и сохранение метрик.
    Включает проверку загружаемых данных и модели.
    """
    try:
        logger.info("Начало оценки модели...")
        
        # Загрузка тестовых данных
        logger.info(f"Загрузка тестовых данных из {X_TEST_PATH} и {Y_TEST_PATH}")
        X_test = pd.read_csv(X_TEST_PATH)
        y_test = pd.read_csv(Y_TEST_PATH)
        
        # Проверка данных
        if X_test.empty or y_test.empty:
            error_msg = "Тестовые данные отсутствуют или пусты"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Загрузка модели
        logger.info(f"Загрузка модели из {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
            
        # Проверка модели
        if not hasattr(model, 'predict'):
            error_msg = "Загруженный объект не является моделью"
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        # Предсказание
        logger.info("Выполнение предсказаний на тестовых данных...")
        y_pred = model.predict(X_test)
        
        # Расчет метрик
        logger.info("Расчет метрик...")
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        # Сохранение метрик
        with open(METRICS_PATH, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        logger.info(f"Метрики успешно сохранены в {METRICS_PATH}")
        logger.info(f"Результаты метрик: {metrics}")
        
    except Exception as e:
        logger.exception(f"Ошибка при оценке модели: {str(e)}")
        raise