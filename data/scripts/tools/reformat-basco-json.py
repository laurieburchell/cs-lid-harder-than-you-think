import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("in_file")
parser.add_argument("out_file")
args = parser.parse_args()

with open(args.in_file, 'r') as f:
    raw_json = json.load(f)

out_data = []
# monolingual sents
for x in raw_json:
    sent = x['referent']
    if x['source_lang']=='es':
        lang = 'spa_Latn'
    elif x['source_lang']=='eu':
        lang = 'eus_Latn'
    out_data.append((sent, lang))

# code-switched sents
cs = [(y['text'], 'spa_Latn', 'eus_Latn') for x in raw_json for y in x['code-switching']]
out_data+=cs

with open(args.out_file, 'w') as f:
    for line in out_data:
        for entry in line:
            f.write(f"{entry}\t")
        f.write('\n')