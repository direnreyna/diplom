# main_train.py

import os
import json
import joblib
import shutil
import torch
import torch.nn as nn
import numpy as np

from src.config import MAX_WORDS, MAX_LEN, STAT_MODEL_PATH, VECTORIZER, MLB, STAT_MODEL, NN_MODEL_PATH, NN_MLB
from src.config import PROJECT_ROOT, INPUT_DIR, TEMP_DIR, DATASET_PATH, MIN_DATASET_DATE

from src.docx_parser import FieldExtractor
from src.file_preparer import FilePreparer
from src.dataset_builder import DatasetBuilder
from src.dataset_manager import DatasetManager
from src.dataset_processor import DatasetProcessor
from src.stat_model_trainer import StatisticalModelTrainer

from datasets import Dataset
from typing import List, Dict
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer


# Глобальные переменные
choosen_model = 'stat'          # stat / nn - выбор обучаемой/применяемой модели.
shared_processor = None         # для хранения процессора
X_train_common = None           # для хранения датасета
y_train_common = None           # для хранения меток
vectorizer = None               # для хранения векторизатора текста
mlb = None                      # для хранения бинаризатора меток
stat_model = None               # для хранения стат модели
exam_is_done = None             # Экзамен в УИИ сдан? Пропуск дальнейших доработок

# Проверяем наличие обученной СтатМодели.
os.makedirs(STAT_MODEL_PATH, exist_ok=True)
# если на диске есть обученная СМ, то загружаем ее...
if os.path.exists(VECTORIZER) and os.path.exists(MLB) and os.path.exists(STAT_MODEL):
    vectorizer = joblib.load(VECTORIZER)
    mlb = joblib.load(MLB)
    stat_model = StatisticalModelTrainer()
    stat_model.load(STAT_MODEL)
    print("✅ Артефакты и модель успешно загружены с диска: {STAT_MODEL_PATH}")
# ...иначе будем обучать
else:
    print(f"❌ Не найдена сохраненная модель. Обучите модель.")

if __name__ == "__main__":
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
    for file_path in docs_path:
        builder.add_file(file_path) 
    new_dataset = builder.build()
    
    # 3. Управление датасетом при частичном пополнении (обновление, бэкап, сохранение)
    # Будет рализовано после сдачи Экзамена: важная вещь, но не на начальном этапе.
    updated_dataset = new_dataset
    manager = DatasetManager(DATASET_PATH)
    manager.backup_dataset()
    
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
        
    # 4. Фильтрация данных (дата / редкие тематики) под Статистическую модель
    filtered_texts_list, filtered_topics_list = manager.prepare_data(updated_dataset, min_date=MIN_DATASET_DATE)

    #####################################################################
    # СТАТИСТИЧЕСКАЯ МОДЕЛЬ LogisticRegression()
    #####################################################################
    if choosen_model == 'stat':
        #####################################################################
        # IV. Векторизация датасета
        #####################################################################
        # 5. Загрузка модели с диска
        # если на диске нет обученной СМ, то обучаем ее...
        if not (os.path.exists(VECTORIZER) and os.path.exists(MLB) and os.path.exists(STAT_MODEL)):
            print(f"❌ Не найдена сохраненная модель. Обучаем модель...")
        
            # Проверяем наличие обученной СтатМодели.
            os.makedirs(STAT_MODEL_PATH, exist_ok=True)

            vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                lowercase=True
            )
            mlb = MultiLabelBinarizer()
            
            # 6. Векторизация данных
            processor = DatasetProcessor(vectorizer, mlb)
            X_train_common, y_train_common, vectorizer, mlb = processor.prepare_model(filtered_texts_list, filtered_topics_list)

            # 7. Разделение на train/test
            X_train, X_test, y_train, y_test = processor.split_dataset(X_train_common, y_train_common)
            print(f"Количество документов: {len(updated_dataset)}.")
            print(f"После фильтрации по дате: { X_train_common.shape[0]} (train={len(X_train.shape[0])}, test={len(X_test.shape[0])})")
            print(f"Количество уникальных тематик: {y_train_common.shape[1]} (train={len(y_train.shape[0])}, test={len(y_test.shape[0])})")

        #####################################################################
        # IV. Обучение модели
        #####################################################################
            # 8. Обучение модели
            trainer = StatisticalModelTrainer()
            trainer.train(X_train, y_train)

            # 9. Сохраняем модель, vectorizer и mlb
            trainer.save("model/statistical_model.pkl")
            joblib.dump(vectorizer, VECTORIZER)
            joblib.dump(mlb, MLB)

            # 10. Оценка
            print("Оценка модели:")
            trainer.evaluate(X_test, y_test, target_names=processor.mlb.classes_)
        
        # ...иначе будем загружать
        else:
            vectorizer = joblib.load(VECTORIZER)
            mlb = joblib.load(MLB)
            stat_model = StatisticalModelTrainer()
            stat_model.load(STAT_MODEL)
            print("✅ Артефакты и модель успешно загружены с диска: {STAT_MODEL_PATH}")

    else:
    #####################################################################
    # 5. Дообучение предобученной модели на базе русскоязычной например 
    # DeepPavlov/rubert-base-cased
    #####################################################################

        # Бинаризация меток полностю как у стат модели
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(filtered_topics_list)
        joblib.dump(mlb, NN_MLB)

        # Токенизация текстов
        tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        # Токенизируем тексты с ограничением длины:
        tokenized_inputs = tokenizer(
            filtered_texts_list,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        # Создание датасета в формате HuggingFace
        dataset = Dataset.from_dict({
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": y
        })

        # Разделение на train/val/test
        train_test_dataset = dataset.train_test_split(test_size=0.2)
        test_val_split = train_test_dataset['test'].train_test_split(test_size=0.5)
        
        #  Можно улучшить: добавить .with_format("torch"), чтобы батчи быстрее читались


        final_dataset = {
            'train': train_test_dataset['train'],
            'val': test_val_split['test'],
            'test': test_val_split['train']
        }
        # Загрузка модели под мультилейбл
        model = AutoModelForSequenceClassification.from_pretrained(
            "DeepPavlov/rubert-base-cased",
            num_labels=y.shape[1],
            problem_type="multi_label_classification",
            id2label={i: label for i, label in enumerate(mlb.classes_)},
            label2id={label: i for i, label in enumerate(mlb.classes_)}
        )

        # Переопределение функции потерь в классе, так как у нас многометочная классификация!
        class MultiLabelTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
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
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=100,
            fp16=True,
            load_best_model_at_end=True,
            metric_for_best_model="f1_micro",
        )
        
        # Оценка качества
        def compute_metrics(pred):
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
        
        # Обучение
        trainer = MultiLabelTrainer(
            model=model,
            args=training_args,
            train_dataset=final_dataset["train"],
            eval_dataset=final_dataset["val"],
            compute_metrics=compute_metrics
        )

        trainer.train()

        # Сохранение модели и артефактов
        model.save_pretrained(NN_MODEL_PATH)
        tokenizer.save_pretrained(NN_MODEL_PATH)