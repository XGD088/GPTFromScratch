import os

import pandas as pd
import tiktoken
import torch
from torch.utils.data import Dataset


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer

        # Truncates sequences if they are longer than max_length
        if max_length is not None:
            self.max_length = max_length
            self.encoded_texts = [
                encoded_text[:self.max_length] for encoded_text in self.encoded_texts
            ]
        else:
            self.max_length = self._longest_encoded_length()

        # pad sequences that are shorter than max_length
        self.encoded_texts = [
            encoded_text + [self.pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def _longest_encoded_length(self):
        return max([len(text) for text in self.encoded_texts])

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)


tokenizer = tiktoken.get_encoding("gpt2")

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件的目录
current_dir = os.path.dirname(current_file_path)


train_dataset = SpamDataset(
    csv_file=os.path.join(current_dir, 'train.csv'),
    max_length=None,
    tokenizer=tokenizer
)

val_dataset = SpamDataset(
    csv_file=os.path.join(current_dir, 'validation.csv'),
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
test_dataset = SpamDataset(
    csv_file=os.path.join(current_dir, 'test.csv'),
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
