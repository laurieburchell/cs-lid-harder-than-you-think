#!/bin/bash
set -eo pipefail

# author: laurie
# downloads training and test data and reformats labels
# see the github issues for an alternative way to access some of the data: https://github.com/laurieburchell/cs-lid-harder-than-you-think/issues/1

HOME_DIR=`pwd`
echo "data download script running in ${HOME_DIR}"
mkdir -p ../train-data
mkdir -p ../test-data

# openLID - from https://github.com/laurieburchell/open-lid-dataset
if ! [ -f ../train-data/lid201-data.fasttext.tsv ]; then
	echo "downloading OpenLID training data"
	cd ../train-data
	if ! [ -f  lid201-data.tsv.gz ]; then 
		echo "downloading OpenLID training data"
		wget https://data.statmt.org/lid/lid201-data.tsv.gz
	else
		echo "found lid201-data.tsv.gz, skipping OpenLID download"
	fi
	echo "decompressing data and changing to fastText format"
	pigz -dc lid201-data.tsv.gz | awk -F"\t" '{print"__label__"$2" "$1}' > lid201-data.fasttext.tsv
	cd $HOME_DIR
else
	echo "found lid201-data.fasttext.tsv, skipping OpenLID download"
fi

# flores-200 
if ! [ -f ../test-data/flores200/flores200.devtest.tsv ]; then
	echo "downloading flores200 test data"
	mkdir -p ../test-data/flores200
	cd ../test-data/flores200
	wget https://tinyurl.com/flores200dataset -O flores200_dataset.tar.gz
	tar -xvf flores200_dataset.tar.gz 
	# reformat flores200 into single test file, then remove arb_Latn, aka_Latn, min_Arab
	cd flores200_dataset/devtest
	echo "reformatting flores-200"
	for file in *; do label=${file%.devtest}; \
		cat $file | awk -v label="$label" -F"\t" '{print $1"\t"__label__}'; done \
		| grep -E -v "__label__(arb_Latn|aka_Latn|min_Arab)" > $HOME_DIR/../test-data/flores200/flores200.devtest.tsv
	rm -rf ../../flores200_dataset
	cd $HOME_DIR
else
	echo "found flores200.devtest.tsv, skipping FLORES-200 download"
fi

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

# LinCE data - from https://huggingface.co/datasets/lince
if ! [[ -f ../test-data/lince/lince_spaeng_validation.tsv && -f ../test-data/lince/lince_msaea_validation.tsv ]]; then
	echo "downloading and reformatting LinCE corpus"
	mkdir -p ../test-data/lince
	cd ../test-data/lince
	# can't currently access with datsets library so I have to deal with the parquet files directly fml
	wget https://huggingface.co/datasets/lince/resolve/refs%2Fconvert%2Fparquet/lid_spaeng/validation/0000.parquet?download=true -O lince_spaeng_validation.parquet
	wget https://huggingface.co/datasets/lince/resolve/refs%2Fconvert%2Fparquet/lid_msaea/validation/0000.parquet?download=true -O lince_msaea_validation.parquet
	python $HOME_DIR/tools/reformat-lince.py lince_spaeng_validation.parquet lince_msaea_validation.parquet \
		lince_spaeng_validation.tsv lince_msaea_validation.tsv
	cd $HOME_DIR
else
	echo "found lince_spaeng_validation.tsv and lince_msaea_validation.tsv, skipping LinCE download"
fi

# ASCEND data - from https://huggingface.co/datasets/CAiRE/ASCEND
if ! [ -f ../test-data/ascend/ascend_train.tsv ]; then
	echo "downloading and reformatting ASCEND corpus"
	mkdir -p ../test-data/ascend
	cd ../test-data/ascend
	python $HOME_DIR/tools/download-and-reformat-ascend.py ascend_train.tsv
	cd $HOME_DIR
else
	echo "found ascend_train.tsv, skipping ASCEND download"
fi

# Turkish-English - from http://tools.nlp.itu.edu.tr/Datasets
if ! [ -f ../test-data/itu-tureng/turkish-english.tsv ]; then
	mkdir -p ../test-data/itu-tureng
	cd ../test-data/itu-tureng
	if ! [ -f turkish-english.raw ]; then
		echo "please obtain the turkish-english data and save it as test-data/itu-tureng/turkish-english.raw"
		exit 1
	fi
	echo "found turkish-english.raw. reformatting turkish-english data"
	python $HOME_DIR/tools/reformat-itu-tureng.py turkish-english.raw turkish-english.tsv
	cd $HOME_DIR
else
	echo "found turkish-english.tsv, ITU dataset reformatting done"
fi

# Indonesian-English - from https://aclanthology.org/D19-5554/
if ! [ -f ../test-data/ind-eng/ind-eng.tsv ]; then
	mkdir -p ../test-data/ind-eng
	cd ../test-data/ind-eng
	if ! [ -f 825_Indonesian_English_CodeMixed.csv ]; then
		echo "please obtain the indonesian-english data and save it as test-data/ind-eng/825_Indonesian_English_CodeMixed.csv"
		exit 1
	fi
	echo "found 825_Indonesian_English_CodeMixed.csv. reformatting indonesian-english data"
	python $HOME_DIR/tools/reformat-indeng.py 825_Indonesian_English_CodeMixed.csv ind-eng.tsv
	cd $HOME_DIR
else
	echo "found ind-eng.tsv, ind-eng dataset reformatting done"
fi
