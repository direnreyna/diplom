# app_gradio.py

import os
import time
import torch
import numpy as np
import gradio as gr
import joblib
from tqdm import tqdm
from docx import Document
from pathlib import Path

# Мои модули
from src.docx_parser import FieldExtractor
from src.file_preparer import FilePreparer
from src.dataset_builder import DatasetBuilder
from src.dataset_manager import DatasetManager
from src.stat_model_trainer import StatisticalModelTrainer
from src.config import PROJECT_ROOT, INPUT_DIR, TEMP_DIR, DATASET_PATH, STAT_MODEL_PATH
from src.config import DATASET_DIR, MIN_DATASET_DATE, VECTORIZER, MLB, STAT_MODEL, ALL_FILTERED_TOPICS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

# Глобальные переменные
freg_allowed_topics = dict()    # частотный словарь тематик
X_train = None                  # выборки датасета
X_val = None                    # выборки датасета
X_test = None                   # выборки датасета
y_train = None                  # выборки датасета
y_val = None                    # выборки датасета
y_test = None                   # выборки датасета
filtered_topics_list = None     # для хранения отфильтрованного списка тематик
vectorizer = None               # для хранения векторизатора текста
mlb = None                      # для хранения бинаризатора меток
stat_model = None               # для хранения стат модели

# 2. Проверка доступности видеокарты
print("CUDA available:", torch.cuda.is_available())
print("GPU device:", torch.cuda.get_device_name(0))
torch.cuda.empty_cache()

# Использование GPU, если доступно
device = ''

# Проверка доступности GPU
if torch.cuda.is_available():
    device = "cuda"  # Явно указываем CUDA
    print(f"Еще раз: GPU доступна: {torch.cuda.get_device_name(0)}. Обучение на {device}")
else:
    device = "cpu"  # Явно указываем ЦПУ
    print(f"Еще раз: GPU недоступна. Обучение на {device}")

###### # Проверяем наличие обученной СтатМодели.
###### os.makedirs(STAT_MODEL_PATH, exist_ok=True)
###### # если на диске есть обученная СМ, то загружаем ее...
###### if os.path.exists(VECTORIZER) and os.path.exists(MLB) and os.path.exists(STAT_MODEL):
######     vectorizer = joblib.load(VECTORIZER)
######     mlb = joblib.load(MLB)
######     stat_model = StatisticalModelTrainer()
######     stat_model.load()
######     print("✅ Артефакты и модель успешно загружены с диска: {STAT_MODEL_PATH}")
###### # ...иначе будем обучать
###### else:
######     print(f"❌ Не найдена сохраненная модель. Обучите модель.")

# === User: 1. Обновление списка файлов ===
def update_user_file_list():
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.txt', '.docx'))]
    return gr.update(choices=files)

# Х === Обновление списка файлов ===
def update_file_list():
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.docx', '.txt', '.zip', '.7z', '.tar.gz', '.tgz', '.rar'))]
    return gr.update(choices=files), "\n".join(files)

# === User: 2. Подготовка выбранного файла ===
def process_single_file(docx_filename):
    global freg_allowed_topics

    if not docx_filename.lower().endswith('.docx'):
        return "❌ Выберите .docx файл", "", ""

    preparer = FilePreparer()
    docx_in_temp = preparer._copy_single_file_to_temp(docx_filename)
    converted_file = preparer.convert_single_docx_in_temp(docx_in_temp)

    parser = FieldExtractor(converted_file)
    text = parser._get_document_text()
    topics = parser._get_topics()

    # Загружаем список наиболее популярных тематик, включенных в ALL_FILTERED_TOPICS
    allowed_topics = set()
    try:
        with open(ALL_FILTERED_TOPICS, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() and "[" in line:
                    topic_part = line.split("[")[1].split("]")[0]
                    topic_freq = line.split("[")[1].split("→ ")[1].strip()
                    allowed_topics.add(topic_part)
                    freg_allowed_topics[topic_part] = topic_freq
    except Exception as e:
        print(f"⚠️ Не удалось загрузить {ALL_FILTERED_TOPICS}: {e}")

    # Фильтруем темы по списку разрешённых
    print(f"Нашли!: {topics}")
    filtered_topics = [topic for topic in topics if topic in allowed_topics]
    print(f"Оставили: {filtered_topics}, потому что в {ALL_FILTERED_TOPICS} ровно {len(allowed_topics)} тематик.")
    
    status_file = f"✅ Обработан файл: {converted_file}"
    status_nodel = "\n✅ Статистическая модель готова к предсказанию" \
        if vectorizer is not None and mlb is not None and stat_model is not None \
        else "❌ Не хватает артефактов для предсказания"

    # Возвращаем содержимое .txt
    return (
        status_file,
        status_nodel,
        text,
        ", ".join([t + f"({freg_allowed_topics[t]})" for t in filtered_topics]) if filtered_topics else "❌ Тематики не найдены"
    )
# === Конвертация выбранных файлов ===
def convert_selected_files(filenames):
    if not filenames:
        return "❌ Ничего не выбрано"
    
    preparer = FilePreparer()
    #docs_path = [os.path.join(INPUT_DIR, f) for f in filenames]
    converted = preparer.prepare_files()
    return f"✅ Обработано: {len(converted)} файлов"

def list_input_files():
    files = os.listdir(INPUT_DIR)
    return "\n".join(files)

def list_temp_files():
    """Возвращает список .txt из temp/"""
    files = [f for f in os.listdir(TEMP_DIR) if f.endswith('.txt')]
    return "\n".join(files)

def get_topics_from_file(filename):
    file_path = os.path.join(INPUT_DIR, filename)
    parser = FieldExtractor(file_path=file_path)
    topics = parser._get_topics()
    return ", ".join(topics) if topics else "❌ Тематики не найдены"

def auto_refresh_on_start():
    input_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.docx', '.txt', '.zip', '.7z', '.tar.gz', '.tgz', '.rar'))]
    temp_files = [f for f in os.listdir(TEMP_DIR) if f.endswith('.txt')]
    dataset_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.json')]
    return (
        "\n".join(input_files),
        f"**Файлы в 'input'** (найдено: {len(input_files)})",
        "\n".join(temp_files),
        f"**Файлы в 'temp'** (найдено: {len(temp_files)})",
        "\n".join(temp_files),
        f"**Файлы в 'temp'** (найдено: {len(temp_files)})",
        "\n".join(dataset_files),
        f"**Датасет в .json** (найдено: {len(dataset_files)})"
    )

def run_prepare_files():
    # Создаем экземпляр FilePreparer и запускаем обработку
    preparer = FilePreparer()
    paths = preparer.prepare_files()
    return f"✅ Обработано файлов: {len(paths)}"

def run_prepare_files_with_progress(progress=gr.Progress()):
    from time import sleep
    
    # Шаг 1: копируем файлы
    progress(0.2, desc="📁 Копируем .docx / .txt")
    preparer = FilePreparer()
    preparer._copy_files_to_temp()
    print("📁 Файлы скопированы")

    # Шаг 2: распаковываем архивы
    progress(0.4, desc="📦 Распаковываем архивы")
    preparer._extract_archives()
    print("📦 Архивы распакованы")

    # Шаг 3: конвертируем docx
    progress(0.7, desc="📄 Конвертируем .docx → .txt")
    preparer._convert_all_docx_in_temp()
    print("📄 .DOCX сконвертированы")

    # Шаг 4: удаляем лишнее
    progress(0.9, desc="🧹 Убираем мусор из temp")
    preparer._remove_non_txt()
    print("🧹 Мусор убран")

    # Шаг 5: завершено
    txt_files = [f for f in os.listdir(TEMP_DIR) if f.endswith('.txt')]
    print(f"✅ Готово! Найдено .txt: {len(txt_files)}")
    
    # Шаг 6: обновляем отображение
    return auto_refresh_on_start()  # вернём обновлённые списки input и temp

def build_dataset():
    builder = DatasetBuilder()
    temp_files = [os.path.join(TEMP_DIR, f) for f in os.listdir(TEMP_DIR) if f.endswith('.txt')]
    
    print(f"📄 Файлов для сборки: {len(temp_files)}")
    
    if not temp_files:
        return "❌ Нет .txt файлов в temp/ → нечего собирать"
    # Оборачиваем итерацию в tqdm для прогресс-бара
    for file in tqdm(temp_files, desc="Собираем ДС из .txt в единый .json", unit="файл"):
        builder.add_file(file)
    
    new_dataset = builder.build()
    dataset_size = len(new_dataset)
    
    manager = DatasetManager(DATASET_PATH)
    manager.backup_dataset()
    manager.save_dataset(new_dataset)
    
    return f"✅ Датасет собран! Документов: {dataset_size}"

def prepare_data_for_training(MIN_DATASET_DATE):
    global X_train, X_val, X_test, y_train, y_val, y_test
    manager = DatasetManager(DATASET_PATH)
    dataset = manager.load_dataset()

    # Фильтрация данных (дата / редкие тематики) под Статистическую модель
    filtered_texts_list, filtered_topics_list = manager.prepare_data(dataset, min_date=MIN_DATASET_DATE)
        
    # Разделение на train/val/test
    X_train, X_val, X_test, y_train, y_val, y_test = manager.split_dataset(filtered_texts_list, filtered_topics_list)

    #print(f"Датасет после фильтра: текстов: {len(filtered_texts_list)}, меток: {len(set(topic for topics in filtered_topics_list for topic in topics))}.")
    print(f"Количество документов до фильтрации: {len(dataset)}.")
    print(f"После фильтрации по дате: {len(filtered_texts_list)} (train={len(X_train)}, val={len(X_val)}, test={len(X_test)})")
    print(f"Количество уникальных тематик: {len(filtered_topics_list)} (train={len(y_train)}, val={len(y_val)}, test={len(y_test)})")

    return (
        f"Документов (после фильтров): {len(filtered_texts_list)}\n(train={len(X_train)}, val={len(X_val)}, test={len(X_test)})",
        f"Тематик (после фильтров): {len(set(topic for topics in filtered_topics_list for topic in topics))}"
    )

# Обучение модели (упрощённая версия)
def train_stat_model(progress=gr.Progress()):
    global X_train, X_val, X_test, y_train, y_val, y_test
    os.makedirs(STAT_MODEL_PATH, exist_ok=True)
    trainer = StatisticalModelTrainer()
    if (os.path.exists(VECTORIZER) and os.path.exists(MLB) and os.path.exists(STAT_MODEL)):
        vectorizer, mlb, model = trainer.load()
        print(f"✅ Артефакты и модель успешно загружены с диска: {STAT_MODEL_PATH}")
        return {f"✅ Обученная модель найдена на диске в папке {STAT_MODEL_PATH}. ❌ Для обучения заново - очистите папку."}
    # если на диске нет обученной СМ, то обучаем ее...
    else:
        vectorizer, mlb, model = trainer.vectorizer, trainer.mlb, trainer.model
        print(f"❌ Не найдена сохраненная модель. Обучаем модель...")
        start_time = time.time()            # Замеряем время

        # Векторизация данных
        X_train_vectorized, y_train_binarized, vectorizer, mlb = trainer.vectorize_dataset(X_train, y_train)
        progress(0.1, desc="📚 Данные векторизованы")
        vector_time = time.time() - start_time

        # Обучение модели
        progress(0.3, desc="🧠 Обучаем модель")
        model = trainer.train(X_train_vectorized, y_train_binarized)
        train_time = time.time() - vector_time

        # Сохраняем модель, vectorizer и mlb
        trainer.save(model, vectorizer, mlb)

    # Оценка модели
    progress(0.9, desc="🧠 Валидация модели")
    metrics = trainer.evaluate(X_test, y_test, target_names=trainer.mlb.classes_)

    # Время окончания
    test_time = time.time() - train_time
    metrics["Время векторизации датасета"] = f"{vector_time:.2f} сек"
    metrics["Время обучения СтатМодели"] = f"{train_time:.2f} сек"
    metrics["Время валидации СтатМодели"] = f"{test_time:.2f} сек"
    return metrics

def predict_topics(text):
    global freg_allowed_topics
    os.makedirs(STAT_MODEL_PATH, exist_ok=True)
    if not (os.path.exists(VECTORIZER) and os.path.exists(MLB) and os.path.exists(STAT_MODEL)):
        return (f"❌ Не найдена сохраненная модель. Упс!...")
    trainer = StatisticalModelTrainer()
    stat_model,vectorizer, mlb = trainer.load()
    print(f"✅ Артефакты и модель успешно загружены с диска: {STAT_MODEL_PATH}")

    X = vectorizer.transform([text])
    print("Ненулевых признаков:", X.nnz)
    print(f"❌ Векторизовали Х...Тип={type(X)}")
    
    y_proba = stat_model.predict_proba(X)
    print(f"❌ Предсказали y_proba...y_proba")

    # --- ДОБАВЛЕНИЕ: вывод ТОП-тем по вероятностям ---
    print("ТОП-5 самых вероятных тем:")
    top_indices = np.argsort(y_proba[0])[-5:][::-1]
    for idx in top_indices:
        print(f"{mlb.classes_[idx]} → {y_proba[0][idx]:.6f}")
    # ---

    # --- ДОБАВЛЕНИЕ: проверка разных порогов ---
    thresholds = [0.15, 0.2, 0.25, 0.3, 0.5]
    print("\nПроверяем разные пороги:")
    for threshold in thresholds:
        preds_binary = (y_proba > threshold).astype(int)
        topics = mlb.inverse_transform(preds_binary)
        print(f"threshold={threshold:.3f} → {topics[0]}")
    # ---

    preds_binary = (y_proba > 0.15).astype(int)  # вероятность класса "1"
    print(f"❌ Перевели ответ в вероятности классов...preds_binary")

    # Восстанавливаем метки
    topics = mlb.inverse_transform(preds_binary)
    print(f"❌ Перевели предсказание в тематики...{topics}")

    return ", ".join([t + f"({freg_allowed_topics[t]})" for t in topics[0] ]) if len(topics) else "❌ Тематики не найдены"

def get_file_content(filename):
    file_path = os.path.join(INPUT_DIR, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

#####################################################################################
with gr.Blocks() as demo:
    gr.Markdown("## Тематический анализ документов по простановке рубрик Классификатора")

    #####################################################################################
    with gr.Tab("🧾 Пользователь"):
        gr.Markdown("### 🔍 Предсказание тематик по одному документу")
        with gr.Row():
            with gr.Column():

                # Выбор файла и его содержимое
                user_file_dropdown = gr.Dropdown(
                    label="Выберите .docx-файл",
                    choices=[f for f in os.listdir(INPUT_DIR) if f.endswith('.docx')],
                    interactive=True
                )
                refresh_user_btn = gr.Button("🔄 Обновить список")
                output_topics = gr.Label(label="Тематики из документа")
                preview_box = gr.Textbox(label="Очищенный текст", lines=10, max_lines=20, interactive=False)
                log_file_box = gr.Textbox(label="Статус файла", lines=1, interactive=False)
                log_model_box = gr.Textbox(label="Статус модели", lines=1, interactive=False)

            with gr.Column():
                pred_topics = gr.Label(label="Предсказанные тематики")
                predict_button = gr.Button("🔮 Предсказать тематики", variant="primary")

                # При нажатии на кнопку → запуск predict_topics() по тексту из preview_box
                predict_button.click(fn=predict_topics, inputs=preview_box, outputs=pred_topics)

        # Привязка событий
        # Обновление списка файлов
        refresh_user_btn.click(fn=update_user_file_list, outputs=user_file_dropdown)
        # При выборе файла → запуск process_single_file()
        user_file_dropdown.select(fn=process_single_file, inputs=user_file_dropdown, outputs=[log_file_box, log_model_box, preview_box, output_topics])
        # При нажатии на кнопку "Предсказать"
        predict_button.click(fn=predict_topics, inputs=preview_box, outputs=pred_topics)
        
    #####################################################################################
    with gr.Tab("Админка"):
        # Прогресс-бар и лог действий
        progress_bar = gr.Progress()
        
        #====================================================================================
        with gr.Accordion("🗂️ ПОДГОТОВКА ФАЙЛОВ -> .txt", open=False):
            with gr.Row():
                with gr.Column():
                    input_count = gr.Markdown(f"**Файлы в 'input'** (найдено: ...)")
                    input_list = gr.Textbox(label="", lines=3, max_lines=3, interactive=False)
                with gr.Column():
                    temp_count = gr.Markdown(f"**Файлы в 'temp'** (найдено: ...)")
                    temp_list = gr.Textbox(label="", lines=3, max_lines=3, interactive=False)

            refresh_bttn = gr.Button("🔄 Обновить окна")
            log_output = gr.Textbox(label="Лог действий", lines=5, interactive=False)
            convert_btn = gr.Button("🔄 Запустить обработку .docx / архивов", variant="primary")

        # Привязка к кнопке
        refresh_bttn.click(
            fn=auto_refresh_on_start,
            outputs=[input_list, input_count, temp_list, temp_count]
        )
        # Привязка функций
        convert_btn.click(fn=run_prepare_files_with_progress, outputs=log_output)

        #====================================================================================
        # 2. Сборка нового датасета
        # builder = DatasetBuilder()
        with gr.Accordion("📚 СБОРКА ДАТАСЕТА", open=False):
            with gr.Row():
                with gr.Column():
                    temp_count_for_ds = gr.Markdown(f"**Файлы в 'temp'** (найдено: ...)")
                    temp_list_for_ds = gr.Textbox(label="", lines=3, max_lines=3, interactive=False)
                with gr.Column():
                    dataset_count = gr.Markdown(f"**ДС в .json** (найдено: ...)")
                    dataset_list = gr.Textbox(label="", lines=3, max_lines=3, interactive=False)

            refresh_bttn = gr.Button("🔄 Обновить окна")
            build_dataset_btn = gr.Button("📂 Собрать датасет из temp/", variant="secondary")
            dataset_status = gr.Textbox(label="Статус сборки", lines=3)

        # Привязка к кнопке
        refresh_bttn.click(
            fn=auto_refresh_on_start,
            outputs=[input_list, input_count, temp_list, temp_count, temp_list_for_ds, temp_count_for_ds, dataset_list, dataset_count]
        )
        # Привязка функций
        build_dataset_btn.click(fn=build_dataset, outputs=dataset_status)

        #====================================================================================
        # 3. Подготовка данных и Обучение модели СМ LogisticRegression()
        with gr.Accordion("📚 ВЕКТОРИЗАЦИЯ и ОБУЧЕНИЕ (модели СМ LogisticRegression)", open=True):
            gr.Markdown("### 🔁 Этапы подготовки данных и обучения")

            # === Подготовка данных ===
            with gr.Row():
                with gr.Column():
                    min_date_input = gr.Textbox(label="Минимальная дата (фильтр)", value=MIN_DATASET_DATE, lines=1)
            prepare_btn = gr.Button("📊 Подготовить данные")
            with gr.Row():
                with gr.Column():
                    doc_count_box = gr.Textbox(label="Кол-во документов после фильтрации", interactive=False)
                with gr.Column():
                    topic_count_box = gr.Textbox(label="Кол-во уникальных тематик", interactive=False)
            with gr.Row():
                with gr.Tab("Обучение Статистической модели"):
                    # === Обучение модели ===
                    train_btn = gr.Button("🧠 Обучить модель", variant="primary")
                    metrics_box = gr.JSON(label="Метрики модели")
                with gr.Tab("Обучение НС модели"):
                    doc_count_box2 = gr.Textbox(label="NN", interactive=False)


        # Привязка к кнопке
        prepare_btn.click(
            fn=prepare_data_for_training,
            inputs=min_date_input,
            outputs=[doc_count_box, topic_count_box]
        )
        
        # Привязка функций
        train_btn.click(fn=train_stat_model, inputs=[], outputs=metrics_box)
        #====================================================================================

    # Автозагрузка при старте
    demo.load(
        fn=auto_refresh_on_start,
        outputs=[input_list, input_count, temp_list, temp_count, temp_list_for_ds, temp_count_for_ds, dataset_list, dataset_count]
    )
    #demo.load(fn=prepare_data_for_training, inputs=min_date_input, outputs=[doc_count_box, topic_count_box])
    # --- При нажатии на кнопку ---
    refresh_bttn.click(
        fn=auto_refresh_on_start,
        outputs=[input_list, input_count, temp_list, temp_count, temp_list_for_ds, temp_count_for_ds, dataset_list, dataset_count]
    )
    
demo.launch()