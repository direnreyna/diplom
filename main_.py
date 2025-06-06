import os
import re

FIELD_NAME_PATTERN = re.compile(
    r'^\$\$(\d+)\s+([^\n\r]+?)\n((?:.(?!\$\$|\Z))*)',
    re.DOTALL | re.MULTILINE
)

HIDDEN_TEXT = re.compile(r'\{[^\}]+\}')
WHITESPACE_PATTERN = re.compile(r'\s+')

def clean_text(self, text):
    text = WHITESPACE_PATTERN.sub(' ', text)
    text = HIDDEN_TEXT.sub('', text)
    return text

def get_topics(fields):
    topics = fields.get("Тематика к документу")
    if topics is None:
        print("Тематика не найдена или отсутствует")
        return []

    # Разделяем по переносам строк + чистим и фильтруем
    raw_topics = topics.split('\n')

    # Оставляем только те, которые проходят проверку
    valid_topics = [topic.strip() for topic in raw_topics]
    valid_topics = [topic for topic in valid_topics if topic]  # Удаляем пустые
    valid_topics = [topic for topic in valid_topics if re.fullmatch(r'\d{5}', topic)]

    if not valid_topics:
        print("Все тематики некорректны или отсутствуют после очистки")

    return valid_topics
    
def parse_text(text):
    """Парсит текст документа на поля"""

    matches = re.findall(FIELD_NAME_PATTERN, text)

    fields = {}
    for number, name, value in matches:
        fields[name] = value.strip()
    return fields

file_paths = ["/home/ayner/projects/EducationProject/data/4.txt"]
    #"/home/ayner/projects/EducationProject/data/temp/055756_RLAW087.txt"]
if 0:
    for file_path in file_paths:
        print(f"{'=' * 80}\n{file_path}\n{'=' * 80}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        fields = parse_text(text)

        for i in fields:
            print(f"[{i}] = [{fields[i]}]")
            #print(f"{w}. [{fields[i]}] = [{fields[k]}]")
        print(f"Тема: {get_topics(fields)}")
if 1:
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"{'=' * 80}\n{file_path}\n{'=' * 80}")
        print(text)
        print(f"{'=' * 80}\n{file_path}\n{'=' * 80}")
        print(len(text))
        print(f"{'=' * 80}\n{file_path}\n{'=' * 80}")
        print(text[:-10])

