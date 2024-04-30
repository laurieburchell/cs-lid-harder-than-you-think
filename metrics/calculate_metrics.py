"""Calculate metrics for the predictions of a model on a test set.

usage: python calculate_metrics.py GOLD_FILE PRED_FILE OUT_FILE
"""

import argparse
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, hamming_loss, multilabel_confusion_matrix
from flores201_classes import flores_201_langs
import numpy as np


def calculate_exact_match_ratio(bin_gold_labels, bin_pred_labels):
    return accuracy_score(bin_gold_labels, bin_pred_labels)

def calculate_hamming_loss(bin_gold_labels, bin_pred_labels):
    return hamming_loss(bin_gold_labels, bin_pred_labels)

def calculate_false_positive_rate(bin_gold_labels, bin_pred_labels):
    mcm = multilabel_confusion_matrix(bin_gold_labels, bin_pred_labels, samplewise=False)
    tn = mcm[:, 0, 0]
    tp = mcm[:, 1, 1]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]
    fpr = np.nan_to_num(fp/(fp+tn))  # avoid division by zero errors
    return fpr.mean()  # false positive rate: FP / (FP+TN)

def calculate_classification_report(bin_gold_labels, bin_pred_labels):
    pass

def calculate_num_unique_langs_predicted(bin_gold_labels, bin_pred_labels):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gold_file", help="path to file of tab-separated gold labels (one per line)")
    parser.add_argument("pred_file", help="path to file of predicted labels (one per line)")
    parser.add_argument("out_file", help="where to write metrics")
    parser.add_argument("--franc", help="convert franc predictions and gold labels to conform", action="store_true")
    args = parser.parse_args()

    # read in gold labels
    with open(args.gold_file, 'r') as f:
        gold_labels = [x.strip().replace('__label__', '').split() for x in f.readlines()]
    # read in predicted labels
    with open(args.pred_file, 'r') as f:
        pred_labels = [x.strip().replace('__label__', '').split() for x in f.readlines()]

    if args.franc:
        raise NotImplementedError("franc conversion not implemented yet")

    # fit binariser based on flores_201_langs
    mlb = MultiLabelBinarizer()
    mlb.fit([flores_201_langs])
    flores_label2index = dict(zip(mlb.classes_, range(len(mlb.classes_))))

    # binarise gold and pred labels
    bin_gold_labels = mlb.transform(gold_labels)
    bin_pred_labels = mlb.transform(pred_labels)

    # calculate metrics
    exact_match = calculate_exact_match_ratio(bin_gold_labels, bin_pred_labels)
    hamming_loss = calculate_hamming_loss(bin_gold_labels, bin_pred_labels)
    fpr = calculate_false_positive_rate(bin_gold_labels, bin_pred_labels)




if __name__ == "__main__":
    main()