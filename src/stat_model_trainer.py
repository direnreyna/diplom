# src/stat_model_trainer.py

import joblib
import numpy as np
from typing import Any, Union, Tuple
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, hamming_loss, jaccard_score
from sklearn.model_selection import train_test_split

class StatisticalModelTrainer:
    def __init__(self, model=None):
        if model is None:
            model = OneVsRestClassifier(LogisticRegression(solver='lbfgs', max_iter=500))
            # Рандом форест
            # model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100))
        self.model = model

    def train(self, X_train: Any, y_train: Any) -> None:
        """Обучает модель"""
        print("🚀 Обучение модели...")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: Any, y_test: Any, target_names=None) -> dict:
        """Оценивает качество модели"""
        #y_pred = self.model.predict(X_test)

        y_proba = self.model.predict_proba(X_test)
        preds_binary = (y_proba > 0.3).astype(int)
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
        print(classification_report(y_test_dense, y_pred, target_names=target_names))

        return self._generate_report(y_test, y_pred)
    


    def _generate_report(self, y_true: Any, y_pred: Any) -> dict:
        """Генерирует отчёт по метрикам"""
        report_str = classification_report(y_true, y_pred)  # Текстовый отчёт (str)
        report_dict = classification_report(y_true, y_pred, output_dict=True)  # Словарь
        
        print(report_str)  # Выводим на экран
        return report_dict  # Возвращаем как dict

    def save(self, path: str) -> None:
        """Сохраняет модель на диск"""
        joblib.dump(self.model, path)
        print(f"💾 Модель сохранена в {path}")

    def load(self, path: str) -> None:
        """Загружает модель с диска"""
        self.model = joblib.load(path)
        print(f"📥 Модель загружена из {path}")