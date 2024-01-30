#!/bin/bash
set -eo pipefail

# author: laurie
# downloads training and test data

HOME_DIR=`pwd`
echo "data download script running in ${HOME_DIR}"

mkdir -p ../train-data
mkdir -p ../test-data

# openLID - from https://github.com/laurieburchell/open-lid-dataset
if ! [ -f ../train-data/lid201-data.fasttext.tsv ]; then
	echo "downloading OpenLID training data"
	cd ../train-data
	wget https://data.statmt.org/lid/lid201-data.tsv.gz
	echo "decompressing data and changing to fastText format"
	pigz -dc lid201-data.tsv.gz | awk -F"\t" '{print"__label__"$2" "$1}' > lid201-data.fasttext.tsv
	cd $HOME_DIR
else
	echo "found lid201-data.fasttext.tsv, skipping OpenLID download"
fi

# flores-200 
if ! [ -f ../test-data/flores200/flores200.devtest.fasttext ]; then
	echo "downloading flores200 test data"
	mkdir -p ../test-data/flores200
	cd ../test-data/flores200
	wget https://tinyurl.com/flores200dataset -O flores200_dataset.tar.gz
	tar -xvf flores200_dataset.tar.gz 
	# reformat flores200 into single test file, then remove arb_Latn, aka_Latn, min_Arab
	cd flores200_dataset/devtest
	echo "changing flores200 devtest to fastText format"
	for file in *; do label=${file%.devtest}; \
		cat $file | awk -v label="$label" -F"\t" '{print "__label__"label" "$1}'; done \
		| grep -E -v "__label__(arb_Latn|aka_Latn|min_Arab)" > ../../flores200.devtest.fasttext
	rm -rf ../../flores200_dataset
	cd $HOME_DIR
else
	echo "found flores200.devtest.fasttext, skipping FLORES-200 download"
fi

# code-switching test sets format: sent \t label1 \t label2

# BaSCo corpus - from https://github.com/Vicomtech/BaSCo-Corpus
if ! [ -f ../test-data/basco/basco_data.tsv ]; then
	echo "downloading and reformatting BaSCo corpus"
	mkdir -p ../test-data/basco
	cd ../test-data/basco
	wget https://raw.githubusercontent.com/Vicomtech/BaSCo-Corpus/main/valid_utterances.json -O basco_valid_utterances.json
	python $HOME_DIR/tools/reformat-basco-json.py basco_valid_utterances.json basco_data.tsv
	pigz basco_valid_utterances.json
	cd $HOME_DIR
else
	echo "found basco_data.tsv, skipping BaSCo download"
fi


