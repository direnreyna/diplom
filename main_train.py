# main_train.py

import os
import json
import joblib
import shutil
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from src.config import MAX_WORDS, MAX_LEN
from src.config import PROJECT_ROOT, INPUT_DIR, TEMP_DIR, MIN_DATASET_DATE
from src.config import DATASET_PATH, STAT_MODEL_PATH, STAT_MODEL, VECTORIZER, MLB
from src.config import NN_DATASET_PATH, NN_MODEL_PATH, NN_MODEL, NN_TOKENIZER, NN_MLB

from src.docx_parser import FieldExtractor
from src.file_preparer import FilePreparer
from src.dataset_builder import DatasetBuilder
from src.dataset_manager import DatasetManager
from src.stat_model_trainer import StatisticalModelTrainer

from datasets import Dataset
from typing import List, Dict
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from torch.utils.data import TensorDataset, DataLoader

# Глобальные переменные
train_model = True              # Флаг - быстрый преход к обучению модели (если есть на диске)
choosen_model = 'nn'            # stat / nn - выбор обучаемой/применяемой модели.
need_train = False              # Флаг - нужно ли обучать модель (если нет на диске обученной)
X_train_common = None           # для хранения датасета
y_train_common = None           # для хранения меток
vectorizer = None               # для хранения векторизатора текста
mlb = None                      # для хранения бинаризатора меток
stat_model = None               # для хранения стат модели
exam_is_done = None             # Экзамен в УИИ сдан? Пропуск дальнейших доработок

if __name__ == "__main__":
    if  train_model:
        #####################################################################
        # ПОДГОТОВКА ДАТАСЕТА
        #####################################################################
        #####################################################################
        # I. Подготовка Файлов
        #####################################################################
        # 1. Подготовка текстовых файлов (извлечение из архивов и/или конвертация из .docx)
        
        preparer = FilePreparer()
        docs_path = preparer.prepare_files()   # тут новых уже нет

        #####################################################################
        # II. Извлечение сырых данных
        #####################################################################
        # 2. Сборка нового датасета
        builder = DatasetBuilder()
        
        print('Найдено текстовых файлов для ДС: ', len(docs_path))
        for file_path in tqdm(docs_path, desc="Собираем ДС из .txt в единый .json", unit="файл"):
            builder.add_file(file_path)
        new_dataset = builder.build()
        
        # 3. Управление датасетом при частичном пополнении (обновление, бэкап, сохранение)
        # Будет рализовано после сдачи Экзамена: важная вещь, но не на начальном этапе.
        updated_dataset = new_dataset
        manager = DatasetManager(DATASET_PATH)
        manager.backup_dataset()
    if 1:    
        #####################################################################
        # Обход для обучения и тестирования, елси есть датсет в .json
        #####################################################################
        #### manager = DatasetManager(NN_DATASET_PATH)
        #### if (os.path.exists(NN_DATASET_PATH)):
        ####     updated_dataset = manager.load_dataset()
        #### print(f"✅Загрузил датасет из .json")

        #####################################################################
        # III. Фильтрация датасета
        #####################################################################
        if exam_is_done:
            old_dataset = manager.load_dataset()
            if builder.is_empty():
                updated_dataset = old_dataset
                print("Новых документов нет — используем старый датасет")
            # Если есть новые, то проверяем старые на предмет измененных.
            # Измененные удалить из новых и обновить их копии в старом ДС
            else:
                updated_dataset, updated_new_items = manager.update_dataset(old_dataset, new_dataset)
                print(f"Добавлено или обновлено {len(updated_dataset) - len(old_dataset)} документов")
            manager.save_dataset(updated_dataset)
        else:
            manager.save_dataset(updated_dataset)
        print(f"✅Сохранил датасет в .json")
            
        # 4. Фильтрация данных (дата / редкие тематики) под Статистическую модель
        filtered_texts_list, filtered_topics_list = manager.prepare_data(updated_dataset, min_date=MIN_DATASET_DATE)
        print(f"✅Отфильтровал ДС")
        print(f"{len(filtered_texts_list)}, {len(filtered_topics_list)}")

        # 5. Разделение на train/val/test
        X_train, X_val, X_test, y_train, y_val, y_test = manager.split_dataset(filtered_texts_list, filtered_topics_list)
        print(f"✅Разделил на train/val/test")
        
        print(f"Количество документов до фильтрации: {len(updated_dataset)}.")
        print(f"После фильтрации по дате: {len(filtered_texts_list)} (train={len(X_train)}, val={len(X_val)}, test={len(X_test)})")

        #####################################################################
        # СТАТИСТИЧЕСКАЯ МОДЕЛЬ LogisticRegression()
        #####################################################################
        if choosen_model == 'stat':
            #####################################################################
            # IV. Векторизация датасета
            #####################################################################
            # 5. Загрузка модели с диска...
            os.makedirs(STAT_MODEL_PATH, exist_ok=True)
            trainer = StatisticalModelTrainer()
            if (os.path.exists(VECTORIZER) and os.path.exists(MLB) and os.path.exists(STAT_MODEL)):
                vectorizer, mlb, model = trainer.load()
                print(f"✅ Артефакты и модель успешно загружены с диска: {STAT_MODEL_PATH}")
            # если на диске нет обученной СМ, то обучаем ее...
            else:
                vectorizer, mlb, model = trainer.vectorizer, trainer.mlb, trainer.model
                print(f"❌ Не найдена сохраненная модель. Обучаем модель...")

                # 6. Векторизация данных
                X_train_vectorized, y_train_binarized, vectorizer, mlb = trainer.vectorize_dataset(X_train, y_train)

                # 8. Обучение модели
                model = trainer.train(X_train_vectorized, y_train_binarized)

                # 9. Сохраняем модель, vectorizer и mlb
                trainer.save(model, vectorizer, mlb)
                
            # 10. Оценка
            print("Оценка модели:")
            metrics = trainer.evaluate(X_test, y_test, target_names=trainer.mlb.classes_)
        else:
        #####################################################################
        # 5. НЕЙРОСЕТЕВАЯ МОДЕЛЬ - DeepPavlov/rubert-base-cased
        #####################################################################

            # Бинаризация меток полностю как у стат модели
            nn_mlb = MultiLabelBinarizer()
            y = nn_mlb.fit_transform(filtered_topics_list)
            joblib.dump(nn_mlb, NN_MLB)
            print(f"✅Бинаризировал метки")

            # Токенизация текстов
            tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

            ### # Токенизируем тексты с ограничением длины:
            ### tokenized_inputs = tokenizer(
            ###     X_train,
            ###     padding="max_length",
            ###     truncation=True,
            ###     max_length=64,
            ###     return_tensors="pt"
            ### )
                       
                        

            def batch_tokenize(texts, tokenizer, batch_size=16, max_length=128):
                """
                Токенизирует список текстов порциями.
                Возвращает список словарей: [{'input_ids': ..., 'attention_mask': ...}, ...]
                """
                tokenized_batches = []
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i : i + batch_size]
                    tokenized = tokenizer(
                        batch_texts,
                        padding="max_length",
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt"
                    )
                    tokenized_batches.append(tokenized)
                
                return tokenized_batches

            def batchify_labels(labels, batch_size=16):
                """Разбивает массив меток на батчи"""
                label_batches = []
                for i in range(0, len(labels), batch_size):
                    batch_labels = labels[i : i + batch_size]
                    label_batches.append(batch_labels)
                return label_batches

            # Параметры
            MAX_LENGTH = 128
            BATCH_SIZE_TOKENIZE = 16
            print("🚀 Начинаем токенизацию по батчам...")
            tokenized_batches_train = batch_tokenize(X_train, tokenizer, batch_size=BATCH_SIZE_TOKENIZE, max_length=MAX_LENGTH)
            tokenized_batches_val = batch_tokenize(X_val, tokenizer, batch_size=BATCH_SIZE_TOKENIZE, max_length=MAX_LENGTH)
            tokenized_batches_test = batch_tokenize(X_test, tokenizer, batch_size=BATCH_SIZE_TOKENIZE, max_length=MAX_LENGTH)
            print(f"✅Токенизировал текст")
            
            label_batches_train = batchify_labels(y_train, batch_size=BATCH_SIZE_TOKENIZE)
            label_batches_val = batchify_labels(y_val, batch_size=BATCH_SIZE_TOKENIZE)
            label_batches_test = batchify_labels(y_test, batch_size=BATCH_SIZE_TOKENIZE)
            
            # Склеиваем все токенизированные батчи в один тензор
            input_ids_train = torch.cat([b["input_ids"] for b in tokenized_batches_train])
            input_ids_val = torch.cat([b["input_ids"] for b in tokenized_batches_val])
            input_ids_test = torch.cat([b["input_ids"] for b in tokenized_batches_test])
            attention_mask_train = torch.cat([b["attention_mask"] for b in tokenized_batches_train])
            attention_mask_val = torch.cat([b["attention_mask"] for b in tokenized_batches_val])
            attention_mask_test = torch.cat([b["attention_mask"] for b in tokenized_batches_test])
            labels_tensor_train = torch.tensor(np.array(label_batches_train), dtype=torch.float32)
            labels_tensor_val = torch.tensor(np.array(label_batches_val), dtype=torch.float32)
            labels_tensor_test = torch.tensor(np.array(label_batches_test), dtype=torch.float32)

            # Создаём PyTorch Dataset
            dataset_train = TensorDataset(input_ids_train, attention_mask_train, labels_tensor_train)
            dataset_val = TensorDataset(input_ids_val, attention_mask_val, labels_tensor_val)
            dataset_test = TensorDataset(input_ids_test, attention_mask_test, labels_tensor_test)

            ### # Создаём HuggingFace Dataset
            ### dataset = Dataset.from_dict({
            ###     "input_ids": input_ids,
            ###     "attention_mask": attention_mask,
            ###     "labels": torch.tensor(y)
            ### })
            print(f"✅Собрал датасет")

            # Загрузка модели под мультилейбл
            model = AutoModelForSequenceClassification.from_pretrained(
                "DeepPavlov/rubert-base-cased",
                num_labels=y.shape[1],
                problem_type="multi_label_classification",
                id2label={i: label for i, label in enumerate(nn_mlb.classes_)},
                label2id={label: i for i, label in enumerate(nn_mlb.classes_)}
            )
            print(f"✅Загрузил мультилейбл модель")

            # Переопределение функции потерь в классе, так как у нас многометочная классификация!
            class MultiLabelTrainer(Trainer):
                #def compute_loss(self, model, inputs, return_outputs=False):
                def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                    labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    logits = outputs.logits
                    loss = nn.BCEWithLogitsLoss()(logits, labels.float())
                    return (loss, outputs) if return_outputs else loss

            training_args = TrainingArguments(
                output_dir="./results",
                learning_rate=2e-5,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=32,
                num_train_epochs=5,
                weight_decay=0.01,
                eval_strategy="epoch",
                save_strategy="epoch",
                logging_dir="./logs",
                logging_steps=10,
                report_to="tensorboard",
                fp16=True,
                load_best_model_at_end=True,
                metric_for_best_model="f1_micro",
            )
            
            # Оценка качества
            def compute_metrics(pred):
                print("Вычисляю метрики...")
                labels = pred.label_ids
                probs = torch.sigmoid(torch.tensor(pred.predictions))
                preds = (probs >= 0.3).int()

                f1_micro = f1_score(labels, preds, average="micro")
                precision_micro = precision_score(labels, preds, average="micro")
                recall_micro = recall_score(labels, preds, average="micro")
                hamming = hamming_loss(labels, preds)

                return {
                    "f1_micro": f1_micro,
                    "precision_micro": precision_micro,
                    "recall_micro": recall_micro,
                    "hamming_loss": hamming
                }

            # Проверка доступности GPU
            if torch.cuda.is_available():
                device = "cuda"  # Явно указываем CUDA
                print(f"Еще раз: GPU доступна: {torch.cuda.get_device_name(0)}. Обучение на {device}")
            else:
                device = "cpu"  # Явно указываем ЦПУ
                print(f"Еще раз: GPU недоступна. Обучение на {device}")
            
            print(f"✅Обучение: начало")
            # Обучение
            trainer = MultiLabelTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset_train,
                eval_dataset=dataset_val,
                compute_metrics=compute_metrics
            )

            trainer.train()
            print(f"✅Обучение: завершено")

            # Сохранение модели и артефактов
            model.save_pretrained(NN_MODEL_PATH)
            tokenizer.save_pretrained(NN_MODEL_PATH)
            print(f"✅Сохранена модель")