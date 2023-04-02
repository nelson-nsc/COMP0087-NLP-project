import json
import os
import numpy as np

from transformers import T5TokenizerFast

PWD = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PWD, "../data/")


def main():
    use_mined = False
    model_name = 't5-base'
    tokenizer = T5TokenizerFast.from_pretrained(model_name)

    # 1. Load Data
    with open(os.path.join(DATA_DIR, "conala-train.json"), 'r') as readFile:
        data = json.load(readFile)
    inputs_ = [str(d['rewritten_intent']) for d in data]
    targets_ = [str(d['snippet']) for d in data]

    if use_mined:
        with open(os.path.join(DATA_DIR, "conala-mined.json"), 'r') as readFile:
            data = json.load(readFile)
        inputs_ += [str(d['intent']) for d in data]
        targets_ += [str(d['snippet']) for d in data]

    # 2. Tokenize Inputs
    input_encode = [tokenizer.tokenize(s) for s in inputs_]
    target_encode = [tokenizer.tokenize(s) for s in targets_]

    input_len_ = [len(s) for s in input_encode]
    target_len_ = [len(s) for s in target_encode]

    print(f"Use Mined: {use_mined} | Model Name: {model_name}")
    print(f"Input Len: Avg {np.mean(input_len_)} Std {np.std(input_len_)} Max {max(input_len_)}")
    print(f"Target Len: Avg {np.mean(target_len_)} Std {np.std(target_len_)} Max {max(target_len_)}")


if __name__ == '__main__':
    main()
