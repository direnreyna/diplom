#docx_parser

import re
from .config import HIDDEN_TEXT, FIELD_NAME_PATTERN, WHITESPACE_PATTERN

class FieldExtractor:

    def __init__(self, file_path=None, text=None):
        # Проверка, какой параметр передан: путь к файлу или его содержимое
        if file_path is None and text is None:
            raise ValueError("Укажите либо file_path, либо text")

        self.whole_doc = ''
        self.fields = {}
        
        if text is not None:
            self._parse_text(text)
        else:
            self._parse_file(file_path)

    def text_to_string(self, text):
        if isinstance(text, list):
            text = ' '.join(text)                   # Склеивание списка в строку
        text = re.sub(r'[^\w\s]', '', text)         # Удаление пунктуации
        #text = re.sub(r'\d+', '', text)             # Удаление чисел
        #text = re.sub(r' [\w] ', ' ', text)         # Удаление отдельных букв
        text = re.sub(r'\s+', ' ', text).strip()    # Удаление лишних пробелов
        #text = text.lower()                         # Приведение к нижнему регистру
        #text = re.sub(r' [фз] ', ' ', text)         # Удаление ФЗ
        text = text[:20000]                         # Удаление "хвостов" длинных документов.

        #text = WHITESPACE_PATTERN.sub(' ', text)
        return text

    def tag_off(self, text):
        text = HIDDEN_TEXT.sub('', text)
        return text
    
    def _parse_file(self, file_path):
        """Читает файл и парсит его"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        # Удаляем скрытый текст
        self.whole_doc = self.tag_off(text)
        self._parse_text(self.whole_doc)

    def _parse_text(self, text):
        """Парсит текст документа на поля"""
        matches = re.findall(FIELD_NAME_PATTERN, text)

        self.fields = {}
        for number, name, value in matches:
            self.fields[name] = value.strip()
        topics = self.fields.get("Тематика к документу")

    def _get_document_text(self):
        text = self.fields.get("Текст", [""])
        print(f"333. Объём текста {len(text)}, тип данных text: {type(text)}")
        # Склеиваем текст в 1 строку
        text = self.text_to_string(text)
        print(f"444. Объём текста {len(text)}, тип данных text: {type(text)}")
        return text

    def _get_topics(self):
        topics = self.fields.get("Тематика к документу")

        if topics is None:
            print("Тематика не найдена или отсутствует")
            return []

        # Разделяем по переносам строк + чистим и фильтруем
        print(f"1. Поле Тематика: {topics}")
        raw_topics = topics.split('\n')
        print(f"2. Сырые тематики: {raw_topics}")

        # Оставляем только те, которые проходят проверку
        valid_topics = [topic.strip() for topic in raw_topics]
        print(f"3. Обрезанные тематики: {valid_topics}")
        valid_topics = [topic for topic in valid_topics if topic]  # Удаляем пустые
        print(f"4. Без пустых тематики: {valid_topics}")
        valid_topics = [topic for topic in valid_topics if re.fullmatch(r'\d{5}', topic)]
        print(f"5. 5-значные тематики: {valid_topics}")

        if not valid_topics:
            print("Все тематики некорректны или отсутствуют после очистки")

        return valid_topics
    
    def _get_doc_id(self):
        return self.fields.get("Номер в ИБ", [""])

    def _get_db_index(self):
    
        """
        Парсит db_index из первой строки файла, если она начинается на $$$$$
        Например: $$$$$RLAW087 → RLAW087
        """
        if not self.whole_doc.strip():
            return ""
        pattern = r'\A\${5}([A-Z]+[\d]*)$'
        first_line = self.whole_doc.lstrip().split('\n', 1)[0]
        match = re.fullmatch(pattern, first_line)
        return match.group(1) if match else ''

    def _get_date_modified(self):
        return self.fields.get("Дата ввода/изменения (часть 1)", [""])

    def _get_all_fields(self):
        return self.fields

    def _get_file_info(self):
        return {
            "Номер": self.fields.get("Номер", [""]),
            "Дата": self.fields.get("Дата", [""]),
            "Название документа": self.fields.get("Название документа", [""]),
            #"Принявший орган": self.fields.get("Принявший орган", [""]),
            "Тематика": self._get_topics(),
            "Текст": self._get_document_text()
        }