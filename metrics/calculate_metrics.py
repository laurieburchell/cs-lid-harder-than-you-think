"""Calculate metrics for the predictions of a model on a test set.

usage: python calculate_metrics.py GOLD_FILE PRED_FILE OUT_FILE
"""

import argparse
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, hamming_loss, multilabel_confusion_matrix, classification_report
import numpy as np
import langcodes
from tqdm import tqdm
from flores201_classes import flores_201_langs  # list of languages in flores_201 dataset


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

def calculate_classification_report(bin_gold_labels, bin_pred_labels, mlb):
    report = classification_report(bin_gold_labels, 
                                   bin_pred_labels, 
                                   target_names=mlb.classes_, 
                                   zero_division=0, 
                                   digits=3)
    return report

def calculate_num_unique_langs_predicted(pred_labels):
    return len(set([item for sublist in pred_labels for item in sublist]))


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

    mlb = MultiLabelBinarizer()

    # if franc, need to convert tags to allow comparison
    if args.franc:
        # first standardise flores labels as alpha3 codes
        flores_201_langs_std_set = set([langcodes.Language.get(x).to_alpha3() for x in flores_201_langs])
        # then standardise franc labels as alpha3 codes
        converted_preds = []
        for preds in tqdm(pred_labels, desc='Standardising franc predictions'):
            converted_tags = []
            for tag in preds:
                if tag not in flores_201_langs_std_set:
                    # if not in tags, get closest equivalent
                    closest_match = langcodes.closest_supported_match(tag, flores_201_langs_std_set, max_distance=10)
                    if closest_match:
                        match = closest_match
                    else:
                        match = tag  # append what I have
                else:
                    match = tag # if tag is there already, all good
                converted_tags.append(langcodes.Language.get(match).to_alpha3())
            converted_preds.append(converted_tags)
            
        # binarise gold and pred labels
        mlb.fit([list(flores_201_langs_std_set)])
        converted_gold_labels = [[langcodes.Language.get(x).to_alpha3() for x in y] for y in tqdm(gold_labels, desc='Standardising gold labels')]
        bin_gold_labels = mlb.transform(converted_gold_labels)
        bin_pred_labels = mlb.transform(converted_preds)

        # if not franc, can fit binariser based on flores_201_langs
    else:
        mlb.fit([flores_201_langs])
        # binarise gold and pred labels
        bin_gold_labels = mlb.transform(gold_labels)
        bin_pred_labels = mlb.transform(pred_labels)

    # calculate metrics
    print("Calculating metrics...")
    exact_match = calculate_exact_match_ratio(bin_gold_labels, bin_pred_labels)
    hamming_loss = calculate_hamming_loss(bin_gold_labels, bin_pred_labels)
    fpr = calculate_false_positive_rate(bin_gold_labels, bin_pred_labels)
    report = calculate_classification_report(bin_gold_labels, bin_pred_labels, mlb)
    num_unique_langs_predicted = calculate_num_unique_langs_predicted(pred_labels)

    # write metrics to file
    with open(args.out_file, 'w') as f:
        f.write(f"Exact match ratio: {exact_match}\n")
        f.write(f"Hamming loss: {hamming_loss}\n")
        f.write(f"False positive rate (macro-average): {fpr}\n")
        f.write(f"Number of unique languages predicted: {num_unique_langs_predicted}\n")
        f.write("\n")
        f.write(report)


if __name__ == "__main__":
    main()