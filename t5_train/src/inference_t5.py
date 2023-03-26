# -*- coding: utf-8 -*-

import argparse
import torch
import os
import yaml
import re

from torch.utils.data import DataLoader, SequentialSampler
from typing import Text, Dict, List
from tqdm import tqdm

PWD = os.path.dirname(os.path.abspath(__file__))


def return_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone_model', type=str, default='t5-base',
                        choices=['t5-base', 't5-large'],
                        help='name of the backbone model to use')

    parser.add_argument('--repeat', type=int, default=0)

    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--use_mined', action="store_true")

    return parser.parse_args()


def get_dataloader(
        tokenizer,
        cfg: Dict,
        use_mined: bool = False
):
    from dataset import TextGenerationDataset
    datasets = TextGenerationDataset(
                tokenizer=tokenizer,
                data_type='test',
                use_mined=use_mined,
                max_length=cfg.get('max_length'),
                padding=cfg.get('padding'),
                truncation=cfg.get('truncation')
            )

    test_dataloader = DataLoader(
        datasets,
        sampler=SequentialSampler(datasets),
        batch_size=cfg.get('batch_size'),
        num_workers=0
    )

    return test_dataloader


def load_model(
        model_name: Text,
):
    from transformers import T5TokenizerFast, T5ForConditionalGeneration

    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model


def load_model_state_dict(model, load_file_name: Text):
    if torch.cuda.is_available():
        state_dict = torch.load(load_file_name)['state_dict']
    else:
        state_dict = torch.load(load_file_name, map_location=torch.device('cpu'))['state_dict']

    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict[key.replace("model.", "")] = value
    model.load_state_dict(new_state_dict)
    return model


class Inferencer:
    def __init__(
            self,
            model,
            tokenizer,
            args: argparse.Namespace,
            batch_size: int = 64,
            max_length: int = 128
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.args = args

        self.model.to(self.device)

    def __call__(self, data_loader) -> List:
        outp = []
        for batch in tqdm(data_loader):
            inputs = dict()
            for key, value in batch.items():
                if key == 'labels':
                    continue
                inputs[key] = value.to(self.device)

            max_input_len = torch.sum(inputs['input_ids'] != 0, dim=1).max().item()
            inputs['input_ids'] = inputs['input_ids'][:, :max_input_len].to(self.device)
            inputs['attention_mask'] = inputs['attention_mask'][:, :max_input_len].to(self.device)
            inputs['max_length'] = self.max_length

            with torch.no_grad():
                outp_ = self.model.generate(**inputs,
                                            num_beams=5,
                                            repetition_penalty=2.5,
                                            length_penalty=1.0,
                                            early_stopping=True
                                            )
                for o in outp_:
                    generated = self.tokenizer.decode(o, max_length=self.max_length)
                    generated = self.prepro_generated_sent(generated)
                    outp.append(generated)
        return outp

    @staticmethod
    def prepro_generated_sent(sent: Text) -> Text:
        PREPRO_PATTERN = re.compile('<[/a-zA-Z0-9_]+>')
        return PREPRO_PATTERN.sub(repl='', string=sent).strip()


def main():
    args = return_args()

    with open(os.path.join(PWD, 'config.yaml'), 'r') as readFile:
        config_file = yaml.load(readFile, Loader=yaml.SafeLoader)
    cfg = config_file.get(args.backbone_model)

    print("Loading model...")
    tokenizer, model = load_model(args.backbone_model)

    model_file_name = os.path.join(PWD, f'../model_binary/{str(args.repeat)}/'
                                        f'/{args.backbone_model}-use_mined{args.use_mined}.ckpt')
    if os.path.isfile(model_file_name):
        model = load_model_state_dict(model, model_file_name)
    else:
        raise FileNotFoundError
    model.eval()

    print(f"Model is Loaded!")
    print(f"Name: {args.backbone_model}")

    print("Start Inference...")
    data_loader = get_dataloader(
            tokenizer,
            cfg=cfg,
            use_mined=args.use_mined
    )

    inferencer = Inferencer(
        model,
        tokenizer,
        batch_size=args.batch_size,
        args=args,
        max_length=cfg.get('max_length')
    )
    prediction = inferencer(data_loader)

    # Save the result
    save_dir = os.path.join(PWD, "../result/")
    os.makedirs(save_dir, exist_ok=True)
    save_file_name = f"{args.backbone_model}-use_mined{args.use_mined}-{str(args.repeat)}.txt"

    with open(os.path.join(save_dir, save_file_name), 'w', encoding='utf-8') as saveFile:
        for line in prediction:
            saveFile.write(line + "\n")


if __name__ == '__main__':
    main()
