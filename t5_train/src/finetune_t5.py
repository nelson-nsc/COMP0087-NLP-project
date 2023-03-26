# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import pytorch_lightning as pl

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from typing import Dict, Text
from transformers import AdamW
from collections import OrderedDict
from torch.nn import CrossEntropyLoss

PWD = os.path.dirname(os.path.abspath(__file__))


class Model(pl.LightningModule):

    def __init__(
            self,
            tokenizer,
            model,
            use_mined,
            cfg: Dict,
            args: argparse.Namespace
    ):
        super(Model, self).__init__()
        self.model = model

        train_dataloader, val_dataloader = get_dataloader(
            tokenizer=tokenizer,
            cfg=cfg,
            use_mined=use_mined
        )

        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader

        self.cfg = cfg
        self.args = args

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay_rate": self.cfg.get('weight_decay_rate')
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay_rate": 0.0
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=float(self.cfg.get('learning_rate')),
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        outp = self.model(**batch)
        loss = outp['loss']

        tqdm_dict = {"train_loss": loss}
        output = OrderedDict({
            "loss": loss,
            "progress_bar": tqdm_dict,
            "log": tqdm_dict
        })
        return output

    def validation_step(self, batch, batch_idx):
        outp = self.model(**batch)
        loss = outp['loss']

        output = OrderedDict({
            "val_loss": loss,
        })

        self.log("val_loss", loss)
        return output

    def test_step(self, batch, batch_idx):
        outp = self.model(**batch)
        loss = outp['loss']

        output = OrderedDict({
            "test_loss": loss,
        })

        self.log("test_loss", loss)
        return output

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader


def get_dataloader(
        tokenizer,
        cfg: Dict,
        use_mined: bool = False
):
    from dataset import TextGenerationDataset
    datasets = [
            TextGenerationDataset(
                tokenizer=tokenizer,
                data_type=d_type,
                use_mined=use_mined,
                max_length=cfg.get('max_length'),
                padding=cfg.get('padding'),
                truncation=cfg.get('truncation')
            )
            for d_type in ['train', 'validation']
        ]

    train_dataloader = DataLoader(
        datasets[0],
        sampler=RandomSampler(datasets[0]),
        batch_size=cfg.get('batch_size'),
        num_workers=0,
    )

    val_dataloader = DataLoader(
        datasets[1],
        sampler=SequentialSampler(datasets[1]),
        batch_size=cfg.get('batch_size'),
        num_workers=0,
    )
    return train_dataloader, val_dataloader


def load_model(
        model_name: Text,
):
    from transformers import T5TokenizerFast, T5ForConditionalGeneration

    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model


def return_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone_model', type=str, default='t5-base',
                        choices=['t5-base', 't5-large'],
                        help='name of the backbone model to use')

    parser.add_argument('--repeat', type=int, default=0)

    parser.add_argument('--use_mined', action="store_true")

    parser.add_argument('--n_gpu', type=int, default=0)

    return parser.parse_args()


def main():
    args = return_args()

    with open(os.path.join(PWD, 'config.yaml'), 'r') as readFile:
        config = yaml.load(readFile, yaml.SafeLoader)
    cfg = config.get(args.backbone_model)

    patience = 3
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0,
        patience=patience,
        verbose=True,
        mode="min"
    )

    save_path = os.path.join(PWD, f"../model_binary/{str(args.repeat)}")
    os.makedirs(save_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        filename=f"{args.backbone_model}-use_mined{args.use_mined}",
        verbose=True,
        save_last=False,
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )

    trainer = pl.Trainer(
        gpus=args.n_gpu,
        accelerator="ddp",
        callbacks=[early_stop_callback, checkpoint_callback],
        max_epochs=cfg.get('epochs'),
    )

    tokenizer, clf = load_model(args.backbone_model)

    model = Model(
        tokenizer,
        clf,
        use_mined=args.use_mined,
        cfg=cfg,
        args=args
    )

    trainer.fit(model)
    result = trainer.test()
    print(result)


if __name__ == '__main__':
    main()
