import argparse
import os
import numpy as np
import torch.cuda
import evaluation
import json
import pandas as pd

from tqdm import tqdm


def return_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone_model', type=str, default='t5-base',
                        choices=['t5-base', 't5-large'],
                        help='name of the backbone model to use')

    parser.add_argument('--repeat', type=int, default=0)

    parser.add_argument('--train_option', type=str, default='hq_mined', choices=['hq', 'hq_mined', 'hq_augment'])

    args = parser.parse_args()
    return args


def main():
    args = return_args()

    # 1. load ground truth:
    with open("../data/conala-test.json", 'r', encoding='utf-8') as targetFile:
        data = json.load(targetFile)
    target = [str(d['snippet']) for d in data]

    # 2. Prepare evaluator
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained(args.backbone_model)

    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-nl", use_fast=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = evaluation.CodeGenerationEvaluator(tokenizer, device, smooth_bleu=True)

    result_text = ""
    for beam_size_ in tqdm([1, 3, 5, 7], desc="repetitions"):
        prediction = []
        with open(f"../result/{args.backbone_model}-{args.train_option}-0-beam{str(beam_size_)}.txt",
                'r', encoding='utf-8') as predictFile:
            for line in predictFile.readlines():
                prediction.append(line.strip())
        assert len(target) == len(prediction)

        # get the bleu score of the results
        outp = {}
        # for ref, pred in tqdm(zip(target, prediction), total=len(target)):
        for ref, pred in zip(target, prediction):

            if pred is not None and pred != "":
                if ref is not None and ref != "":
                    metrics = evaluator.evaluate([pred], [ref])
                    for key, value in metrics.items():
                        if outp.get(key):
                            outp[key].append(value)
                        else:
                            outp[key] = [value]

        for key in outp.keys():
            outp[key] = np.mean(outp[key])

        result_text += f"BLEU Score Beam {str(beam_size_)}: {outp['BLEU']}" + "\n"

    with open(f"../result/{args.backbone_model}-{args.train_option}-result.txt", 'w') as saveFile:
        saveFile.write(result_text)


if __name__ == '__main__':
    main()
