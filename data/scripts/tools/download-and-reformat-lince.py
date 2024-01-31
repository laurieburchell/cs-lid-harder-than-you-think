import argparse
import pandas as pd
import numpy as np
from sacremoses import MosesDetokenizer
#from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("spaeng_in_file")
parser.add_argument("msaea_in_file")
parser.add_argument("spaeng_out_file")
parser.add_argument("msaea_out_file")
args = parser.parse_args()

# currently I can't access the LinCE data with the datasets library so we're dealing with the parquet files
def reformat_lince_parquet(in_file, lang1, lang2):
    """Munge lince lid parquet file and return sents and gold sent-level label(s)"""

    md = MosesDetokenizer()
    df = pd.read_parquet(in_file, engine='pyarrow', columns=['words', 'lid'])
    df['sent'] = df['words'].apply(lambda x: md.detokenize(x))

    sent_level_lid_labels = []
    for labels in df['lid']:
        gold_label = []
        if 'lang1' in labels:
            gold_label.append(lang1)
        if 'lang2' in labels:
            gold_label.append(lang2)
        if len(gold_label) > 0:
            sent_level_lid_labels.append(gold_label)
        else:
            sent_level_lid_labels.append(np.nan)
    df['sent_lid'] = sent_level_lid_labels

    return df[df['sent_lid'].notna()]

def write_out_cs_tsv(df, out_file):
    def combine_columns(row):
        return row['sent'] + '\t' + '\t'.join(row['sent_lid'])
    df['out'] = df.apply(combine_columns, axis=1)
    with open(out_file, 'w') as f:
        for line in df['out']:
            f.write(f"{line}\n")
    

spaeng_df = reformat_lince_parquet(args.spaeng_in_file, 'eng_Latn', 'spa_Latn')
msaea_df = reformat_lince_parquet(args.msaea_in_file, 'arb_Arab', 'arz_Arab')

write_out_cs_tsv(spaeng_df, args.spaeng_out_file)
write_out_cs_tsv(msaea_df, args.msaea_out_file)
