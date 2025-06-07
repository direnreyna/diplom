# calculate_loss_weihgts_with_adasyn.py

import torch
import numpy as np
from typing import Optional, List, Tuple
from imblearn.over_sampling import ADASYN
from transformers import AutoTokenizer, BertModel

class CalculateLossWeihgtsWithADASYN():
    def __init__(self):
        self.class_weights: Optional[np.ndarray] = None  # shape = [n_labels]
    
    def calculate_weights(self, X_train: List[str], y_train: np.ndarray) -> torch.Tensor:
        """
        Основной метод: создаёт веса классов через BERT + ADASYN только для train-выборки.

        :param X_train: список текстов из обучающей выборки
        :param y_train: бинаризованные метки (формат MultiLabelBinarizer)
        :return: class_weights_tensor — тензор с весами классов, shape = [n_labels]
        """

        # 1. Загружаем BERT и токенизатор
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        bert_model = BertModel.from_pretrained("DeepPavlov/rubert-base-cased").to(device)

        # 2. Создаём эмбеддинги только для train
        X_train_emb = self._get_embeddings(X_train, tokenizer, bert_model, device=device)

        # 3. Применяем ADASYN к эмбеддингам + меткам
        X_res, y_res = self._apply_adasyn(X_train_emb, y_train)
        
        # 4. Вычисляем веса классов
        class_weights = self._compute_class_weights(y_res)

        # 5. Переводим в тензор
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

        # 6. Освобождаем BERT из памяти
        del bert_model
        torch.cuda.empty_cache()
        
        return class_weights_tensor

    def _get_embeddings(self, texts: List[str], tokenizer, bert_model, batch_size: int = 8, max_length: int = 128) -> np.ndarray:
        """
        Получает CLS-эмбеддинги для списка текстов порциями.

        :param texts: список текстов
        :param tokenizer: BERT-токенизатор
        :param bert_model: BERT-модель
        :param batch_size: размер батча
        :param max_length: максимальная длина текста
        :return: массив эмбеддингов, shape = [n_samples, 768]
        """
        bert_model.eval()
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(bert_model.device)

            with torch.no_grad():
                outputs = bert_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS-вектор

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)
    
    def _apply_adasyn(self, X_train_emb: np.ndarray, y_train_bin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Применяет ADASYN к эмбеддингам train-выборки.

        :param X_train_emb: эмбеддинги train-текстов, shape = [n_samples, 768]
        :param y_train_bin: бинаризованные метки train, shape = [n_samples, n_labels]
        :return: X_res, y_res — сбалансированные данные после ADASYN
        """
        adasyn = ADASYN(sampling_strategy='auto', random_state=42)
        X_res, y_res = adasyn.fit_resample(X_train_emb, y_train_bin)
        return X_res, y_res

    def _compute_class_weights(self, y_res: np.ndarray) -> np.ndarray:
        """
        Вычисляет веса классов на основе сбалансированных данных после ADASYN.

        :param y_res: бинаризованные метки после ADASYN
        :return: class_weights — нормализованные веса классов
        """
        label_counts = np.mean(y_res, axis=0)
        class_weights = 1. / (label_counts + 1e-8)
        class_weights /= class_weights.sum()  # нормализация
        return class_weights