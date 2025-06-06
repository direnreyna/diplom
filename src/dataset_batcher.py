# dataset_batcher.py

import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

class BatchedTextDataset(Dataset):
    def __init__(self, batch_size=8):
        """
        Базовый класс для работы с батчами текстов и меток.
        Может:
        - создавать батчи и сохранять их на диск,
        - или загружать уже сохранённые данные.
        """
        self.batch_size = batch_size
        self.batch_files = []
        self.label_files = []
        self.batches_dir = None
        self.labels_dir = None
        
    @staticmethod
    def batches_exist(batches_dir, labels_dir):
        """
        Проверяет, существуют ли хотя бы один файл batch_*.pt и labels_batch_*.pt в указанных директориях.
        """
        if not os.path.exists(batches_dir) or not os.path.exists(labels_dir):
            return False

        batch_files = [f for f in os.listdir(batches_dir) if f.startswith("batch_") and f.endswith(".pt")]
        label_files = [f for f in os.listdir(labels_dir) if f.startswith("labels_batch_") and f.endswith(".pt")]

        return len(batch_files) > 0 and len(label_files) > 0
    
    def build_text_batches(self, texts, tokenizer, max_length, batches_path):
        """
        Токенизирует список текстов порциями и сохраняет их на диск.
        Каждый батч сохраняется как отдельный .pt файл в batches_path
        """
        os.makedirs(batches_path, exist_ok=True)
        test_for_shape = ''
        test_for_shape_anomaly = ''
        for i in tqdm(range(0, len(texts), self.batch_size), desc=f"Токенизирую батчами по {self.batch_size}", unit=" док."):

            batch_texts = texts[i : i + self.batch_size]
            tokenized = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            ### # PyTorch < 2.6: Сохраняем батч в файл
            ### batch_number = i // batch_size
            ### batch_filename = os.path.join(batches_path, f"batch_{batch_number}.pt")
            ### torch.save(tokenized, batch_filename)

            ## # Убираем batch-dim, если есть, чтобы гарантировать [batch_size, max_length]
            ## input_ids = tokenized["input_ids"].squeeze() if tokenized["input_ids"].dim() == 2 else tokenized["input_ids"]
            ## attention_mask = tokenized["attention_mask"].squeeze() if tokenized["attention_mask"].dim() == 2 else tokenized["attention_mask"]

            # УБИЙЦА БАГА: убираем лишние измерения при сохранении
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            if input_ids.dim() == 2:
                # Если [1, 8, 64] → ожидаем, что dim == 2 → значит, мы их НЕ обработали ранее
                test_for_shape = f"[!] input_ids.dim() == 2, но форма: {input_ids.shape}"
                #print(f"[!] input_ids.dim() == 2, но форма: {input_ids.shape}")
                # Вывод: [!] input_ids.dim() == 2, но форма: torch.Size([1, 8, 64])
            else:
                test_for_shape_anomaly = f"[Anomaly] input_ids.dim() != 2, но форма: {input_ids.shape}"

            ### # PyTorch >= 2.6:Вместо сохранения всего BatchEncoding — сохраняем словарь тензоров
            ### torch.save({
            ###     "input_ids": tokenized["input_ids"],
            ###     "attention_mask": tokenized["attention_mask"]
            ### }, os.path.join(batches_path, f"batch_{i // batch_size}.pt"))
            # Сохраняем как dict с тензорами, без BatchEncoding
            ## torch.save({
            ##     "input_ids": input_ids,
            ##     "attention_mask": attention_mask
            ## }, os.path.join(batches_path, f"batch_{i // batch_size}.pt"))
            torch.save({
                "input_ids": input_ids.squeeze(),  # ← гарантируем [8, 64]
                "attention_mask": attention_mask.squeeze()
            }, os.path.join(batches_path, f"batch_{i // self.batch_size}.pt"))
        print('=================================================================')
        print(test_for_shape)
        print(test_for_shape_anomaly)
        print('=================================================================')

    def build_label_batches(self, labels, batches_path):
        """
        Разбивает массив меток на батчи и сохраняет их на диск.
        Каждый батч сохраняется как .pt файл в указанной директории.
        """
        os.makedirs(batches_path, exist_ok=True)

        for i in tqdm(range(0, len(labels), self.batch_size), desc=f"Бинаризирую тематики батчами по {self.batch_size}", unit=" рубрик."):
            batch_labels = labels[i : i + self.batch_size]
            batch_number = i // self.batch_size
            # Сохраняем батч в файл
            batch_filename = os.path.join(batches_path, f"labels_batch_{batch_number}.pt")
            torch.save(torch.tensor(batch_labels).float(), batch_filename)

    def __getitem__(self, idx : int):
        """
        Загружает один текст из упакованного в батчи датасета и соответствующую ему метку.
        Возвращает словарь:
            {
                'input_ids': tensor(...),
                'attention_mask': tensor(...),
                'labels': tensor(...)
            }
        """
        # Вычисляем по номеру токена в каком батче он лежит
        batch_idx = idx // self.batch_size
        # ...и его номер в батче
        example_idx = idx % self.batch_size

        text_path = self.batch_files[batch_idx]
        label_path = self.label_files[batch_idx]
       
        # Загружаем токены как обычный dict
        tokenized = torch.load(text_path)
        labels = torch.load(label_path).float()  # метки сразу float
        return {
            'input_ids': tokenized["input_ids"][example_idx],
            'attention_mask': tokenized["attention_mask"][example_idx],
            'labels': labels[example_idx]
        }            
        ### # Убираем лишнее измерение, если есть
        ### input_ids = tokenized['input_ids'].squeeze(0) if tokenized['input_ids'].dim() == 2 else tokenized['input_ids']
        ### attention_mask = tokenized['attention_mask'].squeeze(0) if tokenized['attention_mask'].dim() == 2 else tokenized['attention_mask']

        # # Убираем лишние измерения, если есть
        # input_ids = tokenized["input_ids"].squeeze() if tokenized["input_ids"].dim() >= 2 else tokenized["input_ids"]
        # attention_mask = tokenized["attention_mask"].squeeze() if tokenized["attention_mask"].dim() >= 2 else tokenized["attention_mask"]

        # Убираем все лишние измерения (например, [1, 8, 64] → [8, 64])
        # input_ids = tokenized["input_ids"].squeeze()
        # attention_mask = tokenized["attention_mask"].squeeze()
        # 
        # return {
        #     'input_ids': input_ids,
        #     'attention_mask': attention_mask,
        #     'labels': labels
        # }

    def _load_file_list(self, batches_dir : str, labels_dir : str) -> None:
        """
        Сканирует директории с батчами и метками.
        Сохраняет отсортированные списки путей к файлам в self.batch_files и self.label_files.
        """
        if not os.path.exists(batches_dir) or not os.path.exists(labels_dir):
            raise FileNotFoundError(f"Папки не найдены: {batches_dir}, {labels_dir}")

        # Ищем файлы батчей
        batch_files = [f for f in os.listdir(batches_dir) if f.startswith("batch_") and f.endswith(".pt")]
        batch_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # сортировка по номеру батча
        self.batch_files = [os.path.join(batches_dir, f) for f in batch_files]

        # Ищем файлы меток
        label_files = [f for f in os.listdir(labels_dir) if f.startswith("labels_batch_") and f.endswith(".pt")]
        label_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))  # сортировка по номеру батча
        self.label_files = [os.path.join(labels_dir, f) for f in label_files]

        # Проверяем, что количества совпадают
        if len(self.batch_files) != len(self.label_files):
            raise ValueError(
                f"Количество батчей и меток не совпадает: "
                f"{len(self.batch_files)} vs {len(self.label_files)}"
            )

        self.batches_dir = batches_dir
        self.labels_dir = labels_dir

    def __len__(self) -> int:
        """
        Возвращает количество батчей в датасете.
        """
        return len(self.batch_files)
    
    def load_from_disk(self, batches_dir: str, labels_dir: str) -> None:
    #def load_from_disk(self, batches_dir: str, labels_dir: str, batch_size: int) -> None:
        """
        Загружает информацию о батчах с диска.
        Подготавливает датасет к использованию через __getitem__ и DataLoader.
        """
        #self.batch_size = batch_size
        self._load_file_list(batches_dir, labels_dir)

    def pipeline():
        pass