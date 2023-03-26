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

    parser.add_argument('--use_mined', action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = return_args()

    # 1. load files:
    with open("../data/conala-test.json", 'r', encoding='utf-8') as targetFile:
        data = json.load(targetFile)
    target = [str(d['snippet']) for d in data]

    prediction = []
    with open(f"../result/{args.backbone_model}-use_mined{args.use_mined}-{args.repeat}.txt",
              'r', encoding='utf-8') as predictFile:
        for line in predictFile.readlines():
            prediction.append(line.strip())
    assert len(target) == len(prediction)

    # 2. Prepare evaluator
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained(args.backbone_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = evaluation.CodeGenerationEvaluator(tokenizer, device, smooth_bleu=True)

    # get the bleu score of the results
    outp = {}
    for ref, pred in tqdm(zip(target, prediction), total=len(target)):
        if pred is not None and pred != "":
            if ref is not None and ref != "":
                metrics = evaluator.evaluate([pred], [ref])
                for key, value in metrics.items():
                    if outp.get(key):
                        outp[key].append(value)
                    else:
                        outp[key] = [value]

    for key in outp.keys():
        outp[key] = [np.mean(outp[key])]

    outp_df = pd.DataFrame(outp)
    outp_df.to_csv(f"../result/{args.backbone_model}-use_mined{args.use_mined}-{args.repeat}.tsv", sep='\t')


if __name__ == '__main__':
    main()
