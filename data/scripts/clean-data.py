"""Cleans test sets in format 'sent \t labels...', outputting cleaned sentences to stdout

author: laurie
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
parser.add_argument("in_file", help="test file in tab-separated format 'sent label1 label2")
args = parser.parse_args()

sent_cleaner = SentenceClean()

with open(args.in_file) as f:
    for line in f.readlines():
        raw_sent = line.split('\t')[0].strip()
        clean_sent = sent_cleaner(raw_sent)
        sys.stdout.write(f"{clean_sent}\n")




