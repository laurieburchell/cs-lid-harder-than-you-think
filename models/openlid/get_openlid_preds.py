"""Generate multi-label predictions for a test set using OpenLID model

usage: python get_openlid_preds.py MODEL_PATH TEST_FILE OUT_FILE
"""

import fasttext
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model_path", help="path to OpenLID model (.bin)")
parser.add_argument("test_file", help="path to test file of sentences to classify (one per line)")
parser.add_argument("out_file", help="where to write predictions")

args = parser.parse_args()

# read in test file
with open(args.test_file, 'r') as f:
    test_data = [x.strip() for x in f.readlines()]

# load model
model = fasttext.load_model(args.model_path)

# return predictions thresholded at 0.3
with open(args.out_file, 'w') as f:
    for sent in test_data:
        preds = [x.strip().replace('__label__', '') for x in model.predict(sent, k=2, threshold=0.3)[0]]
        f.write('\t'.join(preds) + '\n')


