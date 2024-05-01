# Code-Switched Language Identification is Harder Than You Think

This repository accompanies the paper [Code-Switched Language Identification is Harder Than You Think](https://aclanthology.org/2024.eacl-long.38/) (EACL 2024). Any problems or suggestions, please [raise an issue](https://github.com/laurieburchell/cs-lid-harder-than-you-think/issues/new)!

Before starting, create a conda envirnoment using [requirements.txt](https://github.com/laurieburchell/cs-lid-harder-than-you-think/blob/main/requirements.txt). All scripts are intended to be run in the directory where they are located.

## Downloading and preprocessing the data
[download-data.sh](https://github.com/laurieburchell/cs-lid-harder-than-you-think/blob/main/data/scripts/download-data.sh) downloads and reformats our training and test data. Note that you need to email the authors to access two of the datasets: see appendix A of the paper for details.

The OpenLID training data is very large (>20GB). Make sure you have enough space available.

```bash
cd data/scripts
bash download_data.sh
```

We provide the [clean-data.py](https://github.com/laurieburchell/cs-lid-harder-than-you-think/blob/main/data/scripts/clean-data.py) script to preprocess the data prior to classification.

## Models
We use three models in this paper: 
- OpenLID
- MultiLID
- Franc

Run [download-models.sh](https://github.com/laurieburchell/cs-lid-harder-than-you-think/blob/main/models/download-models.sh) to obtain the pretrained OpenLID and MultiLID models. Franc is accessed using [pyfranc](https://github.com/cyb3rk0tik/pyfranc). Scripts to obtain predictions from each model are located in `models/<model_name>/get_<model_name>_preds.py`.

## Evaluation
Run [calculate_metrics.py](https://github.com/laurieburchell/cs-lid-harder-than-you-think/blob/main/metrics/calculate_metrics.py) to generate the metrics reported in the paper. It expects the gold file and the predicted labels file to have one set of tab-separated labels per line. If the `--franc` argument is given, it will convert the gold and predicted labels to common alpha3 labels for better comparison. 
