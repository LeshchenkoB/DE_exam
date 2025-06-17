import os
from pathlib import Path

# Базовые пути проекта
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
RESULTS_DIR = BASE_DIR / 'results'
LOGS_DIR = BASE_DIR / 'logs'

# Создание директорий при их отсутствии
DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Пути к файлам
RAW_DATA_PATH = DATA_DIR / 'breast_cancer_wisconsin_diagnostic.csv'
X_TRAIN_PATH = PROCESSED_DIR / 'X_train.csv'
X_TEST_PATH = PROCESSED_DIR / 'X_test.csv'
Y_TRAIN_PATH = PROCESSED_DIR / 'y_train.csv'
Y_TEST_PATH = PROCESSED_DIR / 'y_test.csv'
MODEL_PATH = RESULTS_DIR / 'model.pkl'
METRICS_PATH = RESULTS_DIR / 'metrics.json'

# Настройки логирования
LOGGING_CONFIG = {
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'level': 'INFO',
    'filename': LOGS_DIR / 'pipeline.log'
}

# Проверка важных колонок в данных
REQUIRED_COLUMNS = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension',
    'target'
]