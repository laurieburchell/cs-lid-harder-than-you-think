import argparse
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("out_file")
args = parser.parse_args()

dataset = load_dataset("CAiRE/ASCEND", split='train')
sents = dataset['transcription']
label_map = {'mixed': 'eng_Latn\tzho_Hans',
             'zh': 'zho_Hans',
             'en': 'eng_Latn'}
labels = list(map(lambda x: label_map[x], dataset['language']))

with open(args.out_file, 'w') as f:
    for sent, label in zip(sents, labels):
        f.write(f"{sent}\t{label}\n")