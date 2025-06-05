# src/dataset_processor.py

import json
import joblib
from typing import Tuple, Any, List
from .utils import first_date_is_newer
from .config import VECTORIZER, MLB, TOPIC_FREQ_LIMIT

class to_del_DatasetProcessor:
    def __init__(self, vectorizer, mlb):
        self.vectorizer = vectorizer
        self.mlb = mlb
        topics_list = []


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