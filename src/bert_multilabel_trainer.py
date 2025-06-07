# bert_multilabel_trainer.py
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification

class BertMultiLabelTrainer:
    def __init__(self, model_name="DeepPavlov/rubert-base-cased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=...,
            problem_type="multi_label_classification"
        )
        self.id2label = {...}
        self.label2id = {...}
        self.mlb = ...                  # MultiLabelBinarizer

    def train(self, X_train, y_train):
        ...
    
    def evaluate(self, X_test, y_test):
        ...

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        ...

    def load(self, path):
        self.model = BertForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        ...