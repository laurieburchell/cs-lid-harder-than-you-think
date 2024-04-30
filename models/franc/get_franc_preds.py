"""Generate multi-label predictions for a test set using pyfranc model

usage: python get_franc_preds.py MODEL_PATH TEST_FILE OUT_FILE
"""

from pyfranc import franc
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("test_file", help="path to test file of sentences to classify (one per line)")
parser.add_argument("out_file", help="where to write predictions")

args = parser.parse_args()

# read in test file
print('Reading test data from', args.test_file)
with open(args.test_file, 'r') as f:
    test_data = [x.strip() for x in f.readlines()]

# get franc predictions, filter to those with scores > 0.99
with open(args.out_file, 'w') as f:
    for sent in tqdm(test_data, desc='Predicting with franc'):
        preds = [y[0] for y in franc.lang_detect(sent, min_length=1)[:2] if y[1]>0.99]
        f.write('\t'.join(preds) + '\n')

print('Predictions written to', args.out_file)
print('Note these predictions will need to be converted during evalution')
