# config.py

import re
import os

# Пути
SRC_DIR = os.path.dirname(os.path.abspath(__file__))  # → /home/ayner/projects/EducationProject/src
PROJECT_ROOT = os.path.dirname(SRC_DIR)                 # → /home/ayner/projects/EducationProject
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "input")
TEMP_DIR = os.path.join(PROJECT_ROOT, "data", "temp")
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "dataset.json")
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
STAT_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "stat_model")
NN_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "nn_rubert")

# Параметры датасета
MIN_DATASET_DATE = "01.01.2015"

# Параметры Модели
MAX_WORDS = 10_000
MAX_LEN = 512
STAT_MODEL = os.path.join(STAT_MODEL_PATH, "statistical_model.pkl")
VECTORIZER = os.path.join(STAT_MODEL_PATH, "vectorizer.pkl")
MLB = os.path.join(STAT_MODEL_PATH, "mlb.pkl")
NN_MLB = os.path.join(NN_MODEL_PATH, "mlb.pkl")
TOPIC_FREQ_LIMIT = 20 # Предел частоты тематик, по которому будет работать СМ. Тематики встречающиеся реже не войдут в обучение.
 
# Регэкспы для обработки текста
HIDDEN_TEXT = re.compile(r'\{[^\}]+\}')
WHITESPACE_PATTERN = re.compile(r'\s+')
CLEAN_TOPIC_PATTERN = re.compile(r'^\d{5}$')
FIELD_NAME_PATTERN = re.compile(
    r'^\$\$(\d+)\s+([^\n\r]+?)\n((?:.(?!\$\$|\Z))*)',
    re.DOTALL | re.MULTILINE
)