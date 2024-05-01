#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get either multiple or single labels for a list of input sents
Input: file of inputs, one per line
Output: file of output label(s), one per line
"""

import argparse
import torch
from torchtext.data.utils import get_tokenizer
from slowtext import SlowtextClassifier, SlowtextTrainingDataset, SlowtextTestDataset
from slowtext import build_test_dataloader
from tqdm import tqdm
from scipy.special import expit
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("model_path", help="Slowtext model for classifying sentences")
parser.add_argument("train_dataset", help="training data path to get vocab info")
parser.add_argument("infile", help="path to file of sentences to classify")
parser.add_argument("outfile", help="where to write predictions (will append '.top1' and/or '.multilabel')")
parser.add_argument("--top1", help="return top-1 predictions", action="store_true")
parser.add_argument("--multilabel", help="return multilabel predictions", action="store_true")
parser.add_argument("--emb_dim", help="embedding dimension of model", type=int, default=256)
parser.add_argument("--device", help="train model on cpu or gpu", choices=["cpu", "gpu"], default="cpu")
parser.add_argument("--batch_size", help="batch size for dataloader", type=int, default=2048)
args = parser.parse_args()
if not (args.top1 or args.multilabel):
    parser.error("need to specify at least one of {--top1, --multilabel}")

# need to load training data to get vocab information
train_dataset = torch.load(args.train_dataset)
label2index = train_dataset.label2index_dict
index2label = dict(zip(label2index.values(), label2index.keys()))

# load model
model = SlowtextClassifier(train_dataset, args.emb_dim).to(args.device)
model.load_state_dict(torch.load(args.model_path))
model.eval()

# load test data
tokeniser = get_tokenizer("moses")
test_dataset = SlowtextTestDataset(args.infile, tokeniser, train_dataset)
test_dataloader = build_test_dataloader(test_dataset, 
                                        args.device)

# get logits
with torch.no_grad():
    logits = []
    for idx, (sent, offsets) in enumerate(tqdm(test_dataloader, 
                                                  desc="getting predictions")):
        logit = model(sent, offsets).to("cpu")
        logit = logit.numpy()
        logits.extend(logit)
# convert logits to scores
bce_scores = expit(logits)

# return either top-1 or multilabel
if args.top1:
    print("returning top-1 label prediction")
    pred_labels = [f"{index2label[x]}\n" for x in np.argmax(bce_scores, axis=1)]
    out = f"{args.outfile}.top1"  # alter outfilename
    print(f"writing predictions to {out}")
    with open(out, 'w') as f:
        f.writelines(pred_labels)

if args.multilabel:
    print("returning multilabel predictions")
    thresholds = np.max(bce_scores, axis=1) - np.std(bce_scores, axis=1)
    indices_over_threshold = [np.argwhere(x > i).flatten() for x, i in zip(bce_scores, thresholds)]
    # put results into right format
    pred_labels = []
    for line in indices_over_threshold:
        multilabels = [index2label[i] for i in line]
        pred_labels.append(f"{' '.join(multilabels)}\n")
    out = f"{args.outfile}.multilabel"  # alter outfilename
    print(f"writing predictions to {out}")
    with open(out, 'w') as f:
        f.writelines(pred_labels)
