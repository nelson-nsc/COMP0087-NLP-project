import matplotlib.pyplot as plt

# gpt-2 long snippets
gpt2_results = [26.186927064720408,
 24.612248807707598,
 23.003873108134243,
 22.42810483157713,
 19.634138815098698,
 18.773096348891922,
 18.562921873848442,
 23.19419074727803]

marian_results = [37.81887503587737,
 37.409753269504954,
 35.711178050842165,
 34.08758073112004,
 30.829485944845583,
 31.40210217757424,
 29.67194174556461,
 31.25947692686221]

t5_results = [40.4983,
 39.6745,
 38.5322,
 35.6838,
 32.1644,
 28.0051,
 26.3963,
 22.4087]

lengths = [0, 10, 20, 30, 40, 50, 60, 70]

plt.plot(lengths[:-1], gpt2_results[:-1], label = "GPT-2", marker='o')
plt.plot(lengths[:-1], marian_results[:-1], label = "Marian-MT", marker='s')
plt.plot(lengths[:-1], t5_results[:-1], label = "T5-base", marker='*')
plt.legend()
plt.ylabel('BLEU score')
plt.xlabel('minimum number of characters in snippet')
# plt.title('mod')
plt.savefig("./perf_degrad_plot.png")
# plt.show()
#
# plt.close()
# plt.plot(lengths[:-1], results[:-1])
# plt.ylabel('BLEU score')
# plt.xlabel('minimum number of characters in snippet')
# plt.title('GPT-2 and long snippets')
#
# plt.show()


# marian long snippets
results = [37.81887503587737,
 37.409753269504954,
 35.711178050842165,
 34.08758073112004,
 30.829485944845583,
 31.40210217757424,
 29.67194174556461,
 31.25947692686221]


plt.plot(lengths[:-1], results[:-1])
plt.ylabel('BLEU score')
plt.xlabel('minimum number of characters in snippet')
plt.title('Marian and long snippets')

plt.show()

# tempurature vs bleu
results = [21.235075548792068,
 21.418083550365786,
 20.811302175015783,
 20.67548071714807,
 20.407572897606386,
 19.587856746659853]

temps = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]

fig = plt.figure(figsize=(9,5.5))
plt.plot(temps, results)
plt.xlabel('tempurature value')
plt.ylabel('BLEU')
plt.title('Tempurature parameters vs BLEU')

plt.show()



import json

with open("./data/conala-test.json", 'r', encoding='utf-8') as targetFile:
    data = json.load(targetFile)
target = [str(d['snippet']) for d in data]

# 2. Prepare evaluator
import torch
import numpy as np
import src.evaluation as evaluation
from tqdm import tqdm
from transformers import T5TokenizerFast
tokenizer = T5TokenizerFast.from_pretrained('t5-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
evaluator = evaluation.CodeGenerationEvaluator(tokenizer, device, smooth_bleu=True)

prediction = []
with open(f"./result/t5-base-hq_augment-0-beam5.txt",
          'r', encoding='utf-8') as predictFile:
    for line in predictFile.readlines():
        prediction.append(line.strip())
assert len(target) == len(prediction)

# get the bleu score of the results
outp = {}
# for ref, pred in tqdm(zip(target, prediction), total=len(target)):
bleu_score = []
for ref, pred in tqdm(zip(target, prediction), total=len(target)):
    if pred is not None and pred != "":
        if ref is not None and ref != "":
            metrics = evaluator.evaluate([pred], [ref])
            for key, value in metrics.items():
                if outp.get(key):
                    outp[key].append(value)
                else:
                    outp[key] = [value]

lengths = [0, 10, 20, 30, 40, 50, 60, 70]
len_bleu_dict = {l: [] for l in lengths}
yes = lambda x, l: len(x) > l

for i in lengths:
    for j, tgt_ in enumerate(target):
        if yes(tgt_, i):
            len_bleu_dict[i].append(outp["BLEU"][j])

for key, value in len_bleu_dict.items():
    len_bleu_dict[key] = np.mean(value)
