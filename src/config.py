# config.py

import re
import os

# Пути
DATA_FOLDER = os.path.join('/media', 'Cruiser', 'project')
SRC_DIR = os.path.dirname(os.path.abspath(__file__))  # → /home/ayner/projects/EducationProject/src
PROJECT_ROOT = os.path.dirname(SRC_DIR)                 # → /home/ayner/projects/EducationProject
INPUT_DIR = os.path.join(DATA_FOLDER, "data", "input")
TEMP_DIR = os.path.join(DATA_FOLDER, "data", "temp")
DATASET_DIR = os.path.join(DATA_FOLDER, "data", "processed")
DATASET_PATH = os.path.join(DATASET_DIR, "stat_model_dataset.json")
NN_DATASET_PATH = os.path.join(DATASET_DIR, "nn_rbc_dataset.json")
ALL_FILTERED_TOPICS = os.path.join(DATASET_DIR, "all_filtered_topics.txt")
TOPIC_FREQUENCIES = os.path.join(DATASET_DIR, "topic_frequencies.json")

# Параметры датасета
MIN_DATASET_DATE = "01.01.2015"

# Параметры Модели
MAX_WORDS = 10_000
MAX_LEN = 512

MODEL_DIR = os.path.join(DATA_FOLDER, "model")
STAT_MODEL_PATH = os.path.join(MODEL_DIR, "stat_model")
NN_MODEL_PATH = os.path.join(MODEL_DIR, "nn_rubert")

STAT_MODEL = os.path.join(STAT_MODEL_PATH, "statistical_model.pkl")
VECTORIZER = os.path.join(STAT_MODEL_PATH, "vectorizer.pkl")
MLB = os.path.join(STAT_MODEL_PATH, "mlb.pkl")
NN_MODEL = os.path.join(NN_MODEL_PATH, "nn_rbc_model.pkl")
NN_TOKENIZER = os.path.join(NN_MODEL_PATH, "nn_tokenizer.pkl")
NN_MLB = os.path.join(NN_MODEL_PATH, "nn_mlb.pkl")
MODEL_RESULTS = os.path.join(NN_MODEL_PATH, "results")
NN_TRAIN_BATCHES = os.path.join(DATASET_DIR, "train")
NN_TEST_BATCHES = os.path.join(DATASET_DIR, "test")
NN_VAL_BATCHES = os.path.join(DATASET_DIR, "val")
NN_TRAIN_LABELS_BATCHES = os.path.join(DATASET_DIR, "train_labels")
NN_TEST_LABELS_BATCHES = os.path.join(DATASET_DIR, "test_labels")
NN_VAL_LABELS_BATCHES = os.path.join(DATASET_DIR, "val_labels")

MAX_LENGTH = 256            # лина последовательности текста
BATCH_SIZE_TOKENIZE = 16    # Размер обработанного пакета для хранения на диске перед загрузкой в модель
BATCH_SIZE_TO_MODEL = 16    # Размер батча подаваемого в модель
TOPIC_FREQ_LIMIT = 100 # Предел частоты тематик, по которому будет работать СМ. Тематики встречающиеся реже не войдут в обучение.

# Регэкспы для обработки текста
HIDDEN_TEXT = re.compile(r'\{[^\}]+\}')
WHITESPACE_PATTERN = re.compile(r'\s+')
CLEAN_TOPIC_PATTERN = re.compile(r'^\d{5}$')
FIELD_NAME_PATTERN = re.compile(
    r'^\$\$(\d+)\s+([^\n\r]+?)\n((?:.(?!\$\$|\Z))*)',
    re.DOTALL | re.MULTILINE
)