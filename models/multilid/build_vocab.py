"""
Builds vocabulary and dataset from big file ready for training

Usage: python3 build_vocab.py IN_FILE OUT_FILE

"""

import argparse
from torch import save as torch_save
from torchtext.data.utils import get_tokenizer
from slowtext import SlowtextTrainingDataset


# read in input file and output location from args
parser = argparse.ArgumentParser()
parser.add_argument("infile", help="training file in fastText format")
parser.add_argument("outfile", help="where to output dataset")
parser.add_argument("--min_n", help="min length of n-gram", default=2)
parser.add_argument("--max_n", help="max length of n-gram", default=5)
parser.add_argument("--max_freq", help="min count for word to be in vocab",
                    default=1000)
parser.add_argument("--buckets", help="number of buckets for the n-gram hash table",
                    default=1000000)

args = parser.parse_args()

# set constants for building vocab
tokeniser = get_tokenizer("moses")

# build training dataset
print(f"building training datset from {args.infile}")
train_dataset = SlowtextTrainingDataset(args.infile, 
                                        tokeniser, 
                                        args.min_n, 
                                        args.max_n,
                                        args.max_freq, 
                                        args.buckets)
print("dataset built!")
print(f"saving dataset at {args.outfile}")
torch_save(train_dataset, args.outfile)