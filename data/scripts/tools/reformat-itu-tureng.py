import argparse

parser = argparse.ArgumentParser()
parser.add_argument("in_file")
parser.add_argument("out_file")
args = parser.parse_args()

with open(args.in_file) as f:
    raw = [x.strip()for x in f.readlines()]

sent_accumulator = []
sents = []
tag_accumulator = set()
tags = []

for entry in raw:
    # split into word and tag unless no tag (new sent)
    try:
        word, tag = entry.rsplit(' ', 1)
        sent_accumulator.append(word)
        tag_accumulator.add(tag)

    except ValueError:  # not enough values to unpack, writing time
        s = ' '.join(sent_accumulator)  # write sent
        sents.append(s)
        sent_accumulator = []
        tags_out = []  # write tags
        if 't' in tag_accumulator:
            tags_out += ['tur_Latn']
        if 'e' in tag_accumulator:
            tags_out += ['eng_Latn']
            
        tags.append(tags_out)
        tag_accumulator = set()

with open(args.out_file, 'w') as f:
    for s, t in zip(sents, tags):
        joined_tags = '\t'.join(t)
        f.write(f"{s}\t{joined_tags}\n")