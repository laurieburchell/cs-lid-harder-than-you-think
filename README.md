# Code-Switched Language Identification is Harder Than You Think

This repository accompanies the paper [Code-Switched Language Identification is Harder Than You Think](https://aclanthology.org/2024.eacl-long.38/) (EACL 2024). Any problems or suggestions, please [raise an issue](https://github.com/laurieburchell/cs-lid-harder-than-you-think/issues/new)!


## Downloading and preprocessing the data
Create conda env using [requirements.txt](https://github.com/laurieburchell/cs-lid-harder-than-you-think/blob/main/requirements.txt).

[download-data.sh](https://github.com/laurieburchell/cs-lid-harder-than-you-think/blob/main/data/scripts/download-data.sh) downloads and reformats our training and test data. Note that you need to email the authors to access two of the datasets: see appendix A of the paper for details.

The OpenLID training data is very large (>20GB). Make sure you have enough space available.

```bash
cd data/scripts
bash download_data.sh
```

## Models
We use three models in this paper
- OpenLID
- MultiLID
- Franc
