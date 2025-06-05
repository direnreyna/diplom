# src/dataset_manager.py

import os
import json
import shutil
from collections import Counter
from typing import List, Dict, Tuple, Union, Any
from .utils import first_date_is_newer
from .config import TOPIC_FREQ_LIMIT, ALL_FILTERED_TOPICS, TOPIC_FREQUENCIES
from sklearn.model_selection import train_test_split

class DatasetManager:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.backup_path = dataset_path.replace(".json", "_backup.json")

    def backup_dataset(self) -> None:
        """Делает бэкап текущего датасета"""
        if os.path.exists(self.dataset_path):
            shutil.copy(self.dataset_path, self.backup_path)

    def load_dataset(self) -> List[Dict]:
        """Загружает существующий датасет (если есть)"""
        if not os.path.exists(self.dataset_path):
            return []
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def update_dataset(self, old_data: List[Dict], new_data: List[Dict]) -> Tuple[List[Dict], Union[List[Dict], None]]:
        """
        Обновляет датасет по ключам (doc_id, db_index), проверяя дату изменения
        
        Если new_data пустой — возвращаем старый датасет без изменений
        """
        if not new_data:
            print("Новый датасет пуст — используем старый")
            return old_data, None

        print("Обновляем датасет...")
        old_dict = {(item["doc_id"], item["db_index"]): item for item in old_data}
        
        updated_new_items = []
        for new_item in new_data:
            key = (new_item["doc_id"], new_item["db_index"])
            if key in old_dict:
                if first_date_is_newer(new_item["date_modified"], old_dict[key]["date_modified"]):
                    #print(f"Обновляем запись {key}")
                    old_dict[key] = new_item

                #else:
                    #print(f"Старая версия — пропускаем {key}")
            else:
                #print(f"Добавляем новую запись {key}")
                old_dict[key] = new_item
                updated_new_items.append(new_item)
        return list(old_dict.values()), updated_new_items
    
    def save_dataset(self, data_list: List[Dict]) -> None:
        """Сохраняет датасет с автоматическим созданием директории"""
        output_dir = os.path.dirname(self.dataset_path)
        os.makedirs(output_dir, exist_ok=True)

        with open(self.dataset_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)

        print(f"Датасет сохранён в {self.dataset_path}")

    def _filter_by_date(self, data: list, min_date_str: str) -> list:
        """Фильтрует документы по дате"""
        result = []
        for item in data:
            if first_date_is_newer(item["date_modified"], min_date_str):
                result.append(item)
        return result

    def prepare_data(self, raw_data: list, min_date: str = "01.01.2015", mode: str = 'stat') -> Tuple:
        """
        Подготавливает X и y из сырых данных в зависимости от режима

        :param raw_data: список словарей в формате {"text", "topics", ...}
        :param min_date: минимальная дата документа (в формате dd.mm.yyyy)
        :param mode: 'stat' — для статмодели, 'bert' — для RuBERT
        :return: (X_tfidf, y_binary), список текстов, список меток
        """

        # 1. Фильтруем по дате
        filtered_data = self._filter_by_date(raw_data, min_date)
        texts = [item["text"] for item in filtered_data]
        topics_list = [item["topics"] for item in filtered_data]
        
        print("Количество текстов после фильтрации по дате:", len(texts))
        
        # 2. Считаем частоту встречаемости тематик
        all_topics = [topic for topics in topics_list for topic in topics]
        topic_counter = Counter(all_topics)

        # 3. Сохраняем частоты тематик
        with open(TOPIC_FREQUENCIES, "w", encoding="utf-8") as f:
            json.dump(topic_counter, f, ensure_ascii=False, indent=2)

        # Не понятно - зачем нам дублирование логики?
        # 4. Сохраняем уникальные тематики + частоты в all_topics.txt
        with open("all_topics.txt", "w", encoding="utf-8") as f:
            for i, (topic, count) in enumerate(topic_counter.items()):
                f.write(f"{i}. [{topic}] → {count}\n")

        # 5. Опционально: фильтруем редкие тематики
        if True:
            # Создаем множество допустимых тематик (тех тематик, частота которых больше установленного предела)
            frequent_topics = set(topic for topic, count in topic_counter.items() if count >= TOPIC_FREQ_LIMIT)
            
            # Создаем список меток, в котором остаются только тематики из множества допустимых тематик: frequent_topics
            topics_list_after_del = [[topic for topic in topics if topic in frequent_topics] for topics in topics_list]
            
            # Создаем маску, в которой остаются только документы, содержащие хотя бы одну допустимую тематику
            filter_mask = [bool(topics) for topics in topics_list_after_del]

            # 6. Исключаем документы без тематик после фильтрации (только если нужен чистый ДС для статмодели)

            # Фильтруем labels - список тематик, удаляя те документы, где нет допустимых тематик
            filtered_topics_list = [t for t, keep in zip(topics_list_after_del, filter_mask) if keep]
            # Фильтруем датасет - список текстов, удаляя те документы, где нет допустимых тематик
            filtered_texts_list = [txt for txt, keep in zip(texts, filter_mask) if keep]

            # Считаем частоту встречаемости тематик в отфильтрованном наборе
            all_topics = [topic for topics in filtered_topics_list for topic in topics]
            filtered_topic_counter = Counter(all_topics)

            # Сохраняем уникальные тематики + частоты в отфильтрованном наборе в ALL_FILTERED_TOPICS
            with open(ALL_FILTERED_TOPICS, "w", encoding="utf-8") as f:
                for i, (topic, count) in enumerate(filtered_topic_counter.items()):
                    f.write(f"{i}. [{topic}] → {count}\n")
        else:
            filtered_topics_list = topics_list  # оставляем все тематики
            filtered_texts_list = texts  # оставляем все тексты
        print(f"Все уникальные тематики сохранены в all_topics.txt")
        print(f"Все отфильтрованные уникальные тематики сохранены в ALL_FILTERED_TOPICS")
        print(f"Всего допустимых уникальных тематик: {len(frequent_topics)}")
        
        return filtered_texts_list, filtered_topics_list
    
    def split_dataset(self, X: Any, y: Any, test_size:float=0.2, random_state:int=42) -> Tuple:
        """
        Разбивает данные на train и test.
        """
        # Разделение на train/val/test
        X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test
