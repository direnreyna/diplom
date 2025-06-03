# src/dataset_processor.py

import json
import joblib
from typing import Tuple, Any, List
from sklearn.model_selection import train_test_split
from .utils import first_date_is_newer
from .config import VECTORIZER, MLB, TOPIC_FREQ_LIMIT

class DatasetProcessor:
    def __init__(self, vectorizer, mlb):
        self.vectorizer = vectorizer
        self.mlb = mlb
        topics_list = []

    def prepare_model(self, filtered_texts_list, filtered_topics_list, mode: str = 'stat') -> Tuple:
        """
        Векторизует данные в зависимости от режима

        :param raw_data: подготовленные для векторизации словари X и y
        :param mode: 'stat' — для статмодели, 'bert' — для RuBERT
        :return: (X_tfidf, y_binary), список текстов, список меток
        """
        # Векторизация текстов
        X = self.vectorizer.fit_transform(filtered_texts_list)
        # Бинаризация тематик (мультилейбл кодирование)
        y = self.mlb.fit_transform(filtered_topics_list)

        # Сохраняем vectorizer и mlb
        joblib.dump(self.vectorizer, VECTORIZER)
        joblib.dump(self.mlb, MLB)
        print("💾 vectorizer и mlb сохранены в папку 'model'")

        return X, y, self.vectorizer, self.mlb

    def save_artifacts(self, artifact, path: str) -> None:
        """Сохраняет артефакты на диск"""
        joblib.dump(artifact, path)
        print(f"💾 Артефакт сохранён в {path}")

    def get_label_names(self) -> list:
        """Возвращает список всех тематик (меток)"""
        if not hasattr(self.mlb, 'classes_'):
            raise ValueError("Модель ещё не обучена/подготовлена. Вызовите prepare_data() сначала.")
        return self.mlb.classes_.tolist()

    def vectorize_text(self, text: str):
        """Преобразует один текст в TF-IDF вектор"""
        return self.vectorizer.transform([text])

    def binarize_topics(self, topics_list: list):
        """Бинаризует список тематик (для инференса)"""
        return self.mlb.transform(topics_list)
    
    def split_dataset(self, X: Any, y: Any, test_size:float=0.2, random_state:int=42) -> Tuple:
        """
        Разбивает данные на train и test.
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def evaluate(self, X_test, y_test):
        """
        Оценивает модель на тестовых данных.
        """
        if not hasattr(self, 'model'):
            raise ValueError("Модель не обучена")

        y_pred = self.model.predict(X_test)

        metrics = {
            "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "hamming_loss": hamming_loss(y_test, y_pred)
        }

        print(f"Precision (macro): {metrics['precision']:.4f}")
        print(f"Recall (macro): {metrics['recall']:.4f}")
        print(f"F1-score (macro): {metrics['f1']:.4f}")
        print(f"Hamming loss: {metrics['hamming_loss']:.4f}")

        return metrics