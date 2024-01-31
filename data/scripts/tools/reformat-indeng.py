import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("in_file")
parser.add_argument("out_file")
args = parser.parse_args()

def write_out_cs_tsv(df, out_file):
    def combine_columns(row):
        return row['raw_tweet'] + '\t' + '\t'.join(row['sent_lid'])
    df['out'] = df.apply(combine_columns, axis=1)
    with open(out_file, 'w') as f:
        for line in df['out']:
            f.write(f"{line}\n")

data = pd.read_csv(args.in_file, header=0, usecols=['raw_tweet', 'langs'])
sent_level_labels = []
for labels in data['langs']:
    gold_label = []
    if 'id' in labels:
        gold_label.append('ind_Latn')
    if 'en' in labels:
        gold_label.append('eng_Latn')
    if len(gold_label) > 0:
        sent_level_labels.append(gold_label)
    else:
        sent_level_labels.append(np.nan)
data['sent_lid'] = sent_level_labels

write_out_cs_tsv(data, args.out_file)
