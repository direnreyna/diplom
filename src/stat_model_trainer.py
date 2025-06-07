# src/stat_model_trainer.py

import joblib
import numpy as np
from typing import Any, Union, Tuple, Dict
from scipy.sparse import csr_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, hamming_loss, jaccard_score
from sklearn.model_selection import train_test_split
from .config import VECTORIZER, MLB, STAT_MODEL, STAT_MODEL_PATH

class StatisticalModelTrainer:
    def __init__(self):
        """
        Инициализирует: статистическую модель, векторизатор текста, бинаризатор меток
        """
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            lowercase=True
        )
        self.mlb = MultiLabelBinarizer()
        self.model = OneVsRestClassifier(LogisticRegression(solver='lbfgs', max_iter=500))

    def vectorize_dataset(self, X, y) -> Tuple:
        """
        Векторизует данные в зависимости от режима

        :param raw_data: подготовленные для векторизации словари X и y
        :param mode: 'stat' — для статмодели, 'bert' — для RuBERT
        :return: (X_tfidf, y_binary), список текстов, список меток
        """
        # Векторизация текстов
        X_vec = self.vectorizer.fit_transform(X)

        # Проверяем тип меток
        if isinstance(y, np.ndarray) and len(y.shape) == 2:
            print("Метки уже бинаризованы → пропускаю fit_transform")
            y_bin = y
        else:
            print("Делаю бинаризацию меток...")
            y_bin = self.mlb.fit_transform(y)

        # Сохраняем vectorizer и mlb
        joblib.dump(self.vectorizer, VECTORIZER)
        joblib.dump(self.mlb, MLB)
        print("💾 vectorizer и mlb сохранены в папку 'model'")

        return X_vec, y_bin

    def train(self, X_train: Any, y_train: Any) -> Any:
        """Обучает модель"""
        print("🚀 Обучение модели...")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: Any, y_test: Any, threshold: float = 0.3) -> Union [str, Dict]:
        """Оценивает качество модели"""
        #y_pred = self.model.predict(X_test)

        y_proba = self.model.predict_proba(X_test)
        preds_binary = (y_proba > threshold).astype(int)
        y_pred = preds_binary

        # Hamming Loss — доля неверно предсказанных меток
        print(f"Hamming loss: {hamming_loss(y_test, y_pred):.4f}")

        # Jaccard Score — мера совпадения между истинными и предсказанными метками
        print(f"Jaccard score (samples): {jaccard_score(y_test, y_pred, average='samples'):.4f}")

        # Classification report (по каждой метке отдельно)
        try:
            # Попробуем преобразовать в плотный массив для читаемости отчёта
            y_test_dense = y_test.toarray()
        except:
            y_test_dense = y_test

        print("\nClassification Report:")
        print(classification_report(y_test_dense, y_pred, target_names=self.mlb.classes_))

        return self._generate_report(y_test, y_pred)
    
    def _generate_report(self, y_true: Any, y_pred: Any) -> Union [str, Dict]:
        """Генерирует отчёт по метрикам"""
        report_str = classification_report(y_true, y_pred)  # Текстовый отчёт (str)
        report_dict = classification_report(y_true, y_pred, output_dict=True)  # Словарь
        
        print(report_str)  # Выводим на экран
        return report_dict  # Возвращаем как dict

    def save(self) -> None:
        """Сохраняет модель и артефакты на диск"""
        joblib.dump(self.model, STAT_MODEL)
        joblib.dump(self.vectorizer, VECTORIZER)
        joblib.dump(self.mlb, MLB)
        print(f"💾 Модель сохранена в {STAT_MODEL_PATH}")

    def load(self) -> None:
        """Загружает модель с диска"""
        self.model = joblib.load(STAT_MODEL)
        self.vectorizer = joblib.load(VECTORIZER)
        self.mlb = joblib.load(MLB)
        print(f"📥 Модель и артефакты загружены из {STAT_MODEL_PATH}")