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
from src.config import NN_TRAIN_BATCHES, NN_TEST_BATCHES, NN_VAL_BATCHES
from src.config import NN_TRAIN_LABELS_BATCHES, NN_TEST_LABELS_BATCHES, NN_VAL_LABELS_BATCHES
from src.config import MODEL_RESULTS, MAX_LENGTH, BATCH_SIZE_TOKENIZE, BATCH_SIZE_TO_MODEL

from src.docx_parser import FieldExtractor
from src.file_preparer import FilePreparer
from src.dataset_builder import DatasetBuilder
from src.dataset_manager import DatasetManager
from src.dataset_batcher import BatchedTextDataset
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
choosen_dataset_path = ''
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
    if choosen_model == 'stat':
        choosen_dataset_path = DATASET_PATH
    else:
        choosen_dataset_path = NN_DATASET_PATH
    if 1:
        if train_model:
            manager = DatasetManager(choosen_dataset_path)        
            if (os.path.exists(choosen_dataset_path)):
                updated_dataset = manager.load_dataset()
                print(f"✅Загрузил датасет из .json")
            else:
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
                updated_dataset = new_dataset
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
                # 3. Управление датасетом при частичном пополнении (обновление, бэкап, сохранение)
                # Будет рализовано после сдачи Экзамена: важная вещь, но не на начальном этапе.
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

            # Бинаризация меток
            nn_mlb = MultiLabelBinarizer()
            binarized_topics_list = nn_mlb.fit_transform(filtered_topics_list)
            joblib.dump(nn_mlb, NN_MLB)
            print(f"✅Бинаризировал метки")
            
            # 5. Разделение на train/val/test
            X_train, X_val, X_test, y_train, y_val, y_test = manager.split_dataset(filtered_texts_list, binarized_topics_list)
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
                    trainer.load()
                    print(f"✅ Артефакты и модель успешно загружены с диска: {STAT_MODEL_PATH}")
                # если на диске нет обученной СМ, то обучаем ее...
                else:
                    vectorizer, mlb, model = trainer.vectorizer, trainer.mlb, trainer.model
                    print(f"❌ Не найдена сохраненная модель. Обучаем модель...")

                    # 6. Векторизация данных
                    X_train_vectorized, y_train_binarized = trainer.vectorize_dataset(X_train, y_train)

                    # 8. Обучение модели
                    trainer.train(X_train_vectorized, y_train_binarized)

                    # 9. Сохраняем модель, vectorizer и mlb
                    trainer.save()
                    
                # 10. Оценка
                print("Оценка модели:")
                metrics = trainer.evaluate(X_test, y_test)
            else:
            #####################################################################
            # 5. НЕЙРОСЕТЕВАЯ МОДЕЛЬ - DeepPavlov/rubert-base-cased
            #####################################################################
                ### # Бинаризация меток полностю как у стат модели
                ### nn_mlb = MultiLabelBinarizer()
                ### y = nn_mlb.fit_transform(filtered_topics_list)
                ### joblib.dump(nn_mlb, NN_MLB)
                ### print(f"✅Бинаризировал метки")

                # Токенизация текстов
                tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

                # Параметры
                batcher = BatchedTextDataset(BATCH_SIZE_TOKENIZE)

                print("🚀 Начинаем токенизацию по батчам...")
                if BatchedTextDataset.batches_exist(NN_TRAIN_BATCHES, NN_TRAIN_LABELS_BATCHES):
                    print("Батчи трейна уже сохранены — пропускаем...")
                else:
                    print("Токенизирую трейн...")
                    batcher.build_text_batches(X_train, tokenizer, max_length=MAX_LENGTH, batches_path=NN_TRAIN_BATCHES)

                #batcher.build_text_batches(X_train, tokenizer, max_length=MAX_LENGTH, batches_path=NN_TRAIN_BATCHES)
                print("Токенизирую вал...")
                batcher.build_text_batches(X_val, tokenizer, max_length=MAX_LENGTH, batches_path=NN_VAL_BATCHES)
                print("Токенизирую тест...")
                batcher.build_text_batches(X_test, tokenizer, max_length=MAX_LENGTH, batches_path=NN_TEST_BATCHES)
                print(f"✅Токенизировал текст")
                
                print("🚀 Сохраняем метки по батчам...")
                if BatchedTextDataset.batches_exist(NN_TRAIN_BATCHES, NN_TRAIN_LABELS_BATCHES):
                    print("Метки трейна уже сохранены — пропускаем...")
                else:
                    print("Сохраняю метки трейна...")
                    batcher.build_label_batches(y_train, batches_path=NN_TRAIN_LABELS_BATCHES)

                #batcher.build_label_batches(y_train, batches_path=NN_TRAIN_LABELS_BATCHES)
                print("Сохраняю метки вала...")
                batcher.build_label_batches(y_val, batches_path=NN_VAL_LABELS_BATCHES)
                print("Сохраняю метки теста...")
                batcher.build_label_batches(y_test, batches_path=NN_TEST_LABELS_BATCHES)

                print(f"✅ Сохранил метки по батчам")

                # Создаём и загружаем датасет из сохранённых батчей
                train_dataset = BatchedTextDataset(BATCH_SIZE_TOKENIZE)
                train_dataset.load_from_disk(batches_dir=NN_TRAIN_BATCHES, labels_dir=NN_TRAIN_LABELS_BATCHES)
                # После загрузки датасета
                #print("Пример меток из train_dataset[0]:", train_dataset[0]['labels'])

                val_dataset = BatchedTextDataset(BATCH_SIZE_TOKENIZE)
                val_dataset.load_from_disk(batches_dir=NN_VAL_BATCHES, labels_dir=NN_VAL_LABELS_BATCHES)
                # После загрузки датасета
                #print("Пример меток из val_dataset[0]:", val_dataset[0]['labels'])

                test_dataset = BatchedTextDataset(BATCH_SIZE_TOKENIZE)
                test_dataset.load_from_disk(batches_dir=NN_TEST_BATCHES, labels_dir=NN_TEST_LABELS_BATCHES)
                # После загрузки датасета
                #print("Пример меток из test_dataset[0]:", test_dataset[0]['labels'])
                
                print(f"✅Собрал датасет")

                # Загрузка модели под мультилейбл
                model = AutoModelForSequenceClassification.from_pretrained(
                    "DeepPavlov/rubert-base-cased",
                    num_labels=binarized_topics_list.shape[1],
                    #num_labels=y.shape[1],
                    problem_type="multi_label_classification",
                    id2label={i: label for i, label in enumerate(nn_mlb.classes_)},
                    label2id={label: i for i, label in enumerate(nn_mlb.classes_)}
                )
                model.to("cuda" if torch.cuda.is_available() else "cpu")
                print(f"✅Загрузил мультилейбл модель")

                # Переопределение функции потерь в классе, так как у нас многометочная классификация!
                class MultiLabelTrainer(Trainer):
                    #def compute_loss(self, model, inputs, return_outputs=False):
                    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                        labels = inputs.pop("labels").float()
                        outputs = model(**inputs)
                        logits = outputs.logits
                        loss = nn.BCEWithLogitsLoss()(logits, labels.float())
                        #labels = labels.float() if labels.dtype == torch.long else labels
                        #loss = nn.BCEWithLogitsLoss()(logits, labels)                    
                        return (loss, outputs) if return_outputs else loss

                training_args = TrainingArguments(
                    output_dir=MODEL_RESULTS,
                    learning_rate=0.001,
                    per_device_train_batch_size=BATCH_SIZE_TO_MODEL,
                    per_device_eval_batch_size=BATCH_SIZE_TO_MODEL,
                    num_train_epochs=50,
                    weight_decay=0.01,
                    eval_strategy="epoch",
                    save_strategy="epoch",
                    logging_dir="./logs",
                    logging_steps=1,
                    report_to="tensorboard",
                    fp16=torch.cuda.is_available(),  
                    load_best_model_at_end=True,
                    metric_for_best_model="f1_micro",
                    remove_unused_columns=False, 
                    torch_compile=True, # optimizations
                    optim="adamw_torch_fused", # improved optimizer
                )
                
                # Оценка качества
                def compute_metrics(pred):
                    print("Вычисляю метрики...")
                    labels = pred.label_ids
                    probs = torch.sigmoid(torch.tensor(pred.predictions))

                    # 🔍 Проверка: какие метки пришли?
                    print("Пример меток (первые 5):", labels[:5].sum(axis=1))  # посмотри, есть ли вообще 1 в метках
                    print("Пример probs (первые 5):", probs[:5].mean(dim=1))   # средняя вероятность активации

                    preds = (probs >= 0.001).int()

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

                ### # Проверка доступности GPU
                ### if torch.cuda.is_available():
                ###     device = "cuda"  # Явно указываем CUDA
                ###     print(f"Еще раз: GPU доступна: {torch.cuda.get_device_name(0)}. Обучение на {device}")
                ### else:
                ###     device = "cpu"  # Явно указываем ЦПУ
                ###     print(f"Еще раз: GPU недоступна. Обучение на {device}")

                print(f"✅Обучение: начало")
                # Обучение
                trainer = MultiLabelTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    compute_metrics=compute_metrics
                )

                trainer.train()
                if 1==0:
                    # Загрузим последний батч из трейна
                    first_batch = train_dataset[len(train_dataset) - 1]
                    print("input_ids shape:", first_batch["input_ids"].shape)
                    print("labels shape:", first_batch["labels"].shape)

                    # Перемещаем все тензоры на устройство модели
                    first_batch = {k: v.to(model.device) for k, v in first_batch.items()}

                    # Убираем unsqueeze, если не нужен (если батч уже правильного размера)
                    outputs = model(**first_batch)
                    logits = outputs.logits

                    # Посчитаем loss
                    loss = nn.BCEWithLogitsLoss()(logits, first_batch['labels'].float())
                    print("Loss:", loss.item())
                                
                print(f"✅Обучение: завершено")

                # Сохранение модели и артефактов
                model.save_pretrained(NN_MODEL_PATH)
                tokenizer.save_pretrained(NN_MODEL_PATH)
                print(f"✅Сохранена модель")

            
                # Оценка на тесте после обучения
                print("🧪 Оцениваю модель на тестовом датасете...")
                print(f"test_dataset={len(test_dataset)}")
                test_results = trainer.evaluate(test_dataset)
                print("Результаты на тесте:", test_results)
                print(f"✅Модель протестирована")
