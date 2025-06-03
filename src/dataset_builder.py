# src/dataset_builder.py

import os
import json
from typing import List, Dict
from .docx_parser import FieldExtractor

class DatasetBuilder:
    def __init__(self):
        self.dataset = []

    def add_file(self, file_path: str) -> None:
        base_file = os.path.basename(file_path)
        """Добавляет документ в датасет"""
        # Парсим данные
        parser = FieldExtractor(file_path=file_path)
        topics = parser._get_topics()
        #print(f"222. {base_file} - Тематики {topics}")
        doc_text = parser._get_document_text()
        doc_id = parser._get_doc_id()
        db_index = parser._get_db_index()
        date_modified = parser._get_date_modified()

        ###################################################################
        ## Проверяем, есть ли уже частотный словарь от прошлого запуска
        #freq_path = "topic_frequencies.json"
        #if os.path.exists(freq_path):
        #    with open(freq_path, "r", encoding="utf-8") as f:
        #        freq_dict = json.load(f)
        #
        #    # Оставляем только те тематики, что встречаются часто
        #    topics = [t for t in topics if freq_dict.get(t, 0) >= 10]
        ###################################################################

        if not topics:
            print(f"❌ Файл без тематик пропущен: {file_path}")
            return

        self.dataset.append({
            "text": doc_text,
            "topics": topics,
            "doc_id": doc_id,
            "db_index": db_index,
            "date_modified": date_modified
        })

    def build(self) -> List[Dict]:
        """Возвращает собранный датасет"""
        return self.dataset
    
    def is_empty(self) -> bool:
        """Возвращает 0 для пустого датасета"""
        return len(self.dataset) == 0