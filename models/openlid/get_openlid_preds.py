"""Generate multi-label predictions for a test set using OpenLID model

usage: python get_openlid_preds.py MODEL_PATH TEST_FILE OUT_FILE
"""

import fasttext
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model_path", help="path to OpenLID model")
parser.add_argument("test_file", help="path to test file of sentences to classify")
parser.add_argument("out_file", help="where to write predictions")

args = parser.parse_args()



