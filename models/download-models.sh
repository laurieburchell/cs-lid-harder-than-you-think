#!/bin/bash
set -eo pipefail

# author: laurie
# downloads models

HOME_DIR=`pwd`

# openlid - from https://github.com/laurieburchell/open-lid-dataset
cd openlid
if ! [ -f lid201-model.bin ]; then
    echo "downloading OpenLID model"
    wget https://data.statmt.org/lid/lid201-model.bin.gz
    pigz -d lid201-model.bin.gz
else
    echo "found lid201-model.bin, skipping OpenLID model download"
fi
cd $HOME_DIR

# multilid - download from University of Edinburgh's server
cd multilid
if ! [ -f multilid-model.pt ]; then
    echo "downloading MultiLID model"
    wget https://data.statmt.org/lid/multilid-model.pt
else
    echo "found multilid-model.pt, skipping MultiLID model download"
fi
cd $HOME_DIR

echo "Use pyfranc to access the franc-all model"