"""Cleans test sets in format 'sent \t labels...', outputting cleaned sentences to a file

author: laurie
Usage : python clean-data.py <input_file> [--out_file <output_file>] [--clean_file <cleaned_sentences_file>]
"""

import argparse
import sys
import unicodedata
from sacremoses import MosesPunctNormalizer
from tools.defines import Patterns
from tools.remove_non_printing_char import get_replacer as non_printing_char_replacer
from tools.demojizier import Demojizer

class SentenceClean:
    def __init__(self):
        self.mpn = MosesPunctNormalizer(lang='en')
        self.replace_nonprint = non_printing_char_replacer(" ")
        self.demojiser = Demojizer()

    def __call__(self, line):
        clean = self.mpn.normalize(line)
        clean = self.replace_nonprint(clean)
        clean = unicodedata.normalize("NFKC", clean)
        clean = self.demojiser(clean, "")

        # remove twitter effects
        clean = Patterns.URL_PATTERN.sub('', clean)
        clean = Patterns.HASHTAG_PATTERN.sub('', clean)
        clean = Patterns.MENTION_PATTERN.sub('', clean)
        clean = Patterns.RESERVED_WORDS_PATTERN.sub('', clean)
        clean = Patterns.NUMBERS_PATTERN.sub('', clean)

        return clean

parser = argparse.ArgumentParser()
parser.add_argument(
    "in_file", 
    type=str, 
    help="Path to the input file in tab-separated format 'sent label1 label2'"
)

parser.add_argument(
    "--out_file",
    type=str,
    default="gold_labels_ascend.txt",
    help="Path to the output file for gold labels"
)

parser.add_argument(
    "--clean_file",
    type=str,
    default="cleaned_ascend.txt",
    help="Path to the output file for cleaned sentences"
)

args = parser.parse_args()

if not args.in_file:
    parser.error("The input file path is required. Please provide it as an argument.")

sent_cleaner = SentenceClean()

with open(args.in_file, 'r', encoding='utf-8') as f, \
     open(args.out_file, 'w', encoding='utf-8') as gold_out, \
     open(args.clean_file, 'w', encoding='utf-8') as clean_out:
    for line in f.readlines():
        parts = line.strip().split('\t')
        if len(parts) > 1:
            raw_sent = parts[0].strip()
            labels = parts[1:]  # Extract labels
            clean_sent = sent_cleaner(raw_sent)
            clean_out.write(f"{clean_sent}\n")  # Write cleaned sentences to cleaned_sentences.txt
            gold_out.write("\t".join(labels) + "\n")  # Write labels to gold_labels.txt
