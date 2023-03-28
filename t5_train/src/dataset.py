import json

import pandas as pd
import torch
import os
import numpy as np

from typing import Text
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

PWD = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PWD, "../data/")


class TextGenerationDataset(Dataset):

    def __init__(
            self,
            tokenizer,
            data_type: Text,
            train_option: Text = 'hq',
            seed: int = 1234,
            max_length: int = 128,
            padding: Text = "max_length",
            truncation: Text = "longest_first"
    ):
        assert data_type in ['train', 'validation', 'test']
        assert train_option in ['hq', 'hq_mined', 'hq_augment']
        self.cross_entropy_ignore_index = -100

        # 1. Load Data
        print("Loading Data...")
        if data_type == 'test':
            with open(os.path.join(DATA_DIR, "conala-test.json"), 'r') as readFile:
                data = json.load(readFile)
            inputs_ = [str(d['rewritten_intent']) for d in data]
            targets_ = [str(d['snippet']) for d in data]
        elif data_type == 'validation':
            data = pd.read_csv(os.path.join(DATA_DIR, "hq_val2.csv"), sep=',')
            inputs_ = data['intent'].tolist()
            targets_ = data['snippet'].tolist()
        else:
            data = pd.read_csv(os.path.join(DATA_DIR, "hq_train2.csv"), sep=',')
            inputs_ = data['intent'].tolist()
            targets_ = data['snippet'].tolist()

            if train_option == 'hq_mined':
                data = []
                with open(os.path.join(DATA_DIR, "conala-mined.jsonl"), 'r') as readFile:
                    for line in readFile.readlines():
                        data.append(json.loads(line))

                import random
                random.seed(seed)
                random.shuffle(data)
                inputs_ += [str(d['intent']) for d in data[:10000]]
                targets_ += [str(d['snippet']) for d in data[:10000]]
            elif train_option == 'hq_augment':
                data = pd.read_csv(os.path.join(DATA_DIR, "train_aug.csv"), sep=',')
                inputs_ += data['intent'].tolist()
                targets_ += data['snippet'].tolist()
            else:
                pass

        # 2. Tokenize Inputs
        print("Tokenizing Data...")
        input_encode = tokenizer(inputs_, max_length=max_length, truncation=truncation, padding=padding)
        target_encode = tokenizer(targets_, max_length=max_length, truncation=truncation, padding=padding)

        self.outputs = dict()
        self.outputs["labels"] = target_encode['input_ids']
        self.outputs["input_ids"] = input_encode['input_ids']
        self.outputs["attention_mask"] = input_encode['attention_mask']

    def __len__(self):
        return len(self.outputs['input_ids'])

    def __getitem__(self, item):
        outp_item = {key: torch.LongTensor(val[item]) for key, val in self.outputs.items()}
        if 'labels' in list(outp_item.keys()):
            labels = outp_item['labels']
            prompt_length = int(sum(labels != 0))
            outp_item['labels'][prompt_length:] = self.cross_entropy_ignore_index  # Cross Entropy Ignore Index
        return outp_item


if __name__ == '__main__':
    # EDA
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained('t5-base')
    dataset = TextGenerationDataset(
        tokenizer=tokenizer,
        data_type='train',
        use_mined=True
    )

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    for b in loader:
        print(b['labels'].size())