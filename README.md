# Project Summary

* Building a predictive model to continuously forecast customer behaviour and optimize marketing strategies with historical transaction data.
* Implementing tabular variational autoencoders to synthesize transaction records, with item embedding over 2000+ retailers.
* Applying regression models using macroeconomic data to constrain monthly transaction counts, where data is collected with API from Statistics Canada.

## How to use the package

The package includes two scripts. `train.py` uses raw data to train synthesizers. `generate.py` uses synthesizers (trained by `train.py`) to generate transaction records with specified year and month. Both scripts accept key-value pair arguments.

### `train.py`

Inputs: 
  * `--f` or `--filename`, string, raw data file directory, default = `data/cc_data.csv`
  * `--n` or `--samplenum`, integer, number of samples randomly chosen for training, default = `1000`
  * `--s` or `--save`, boolean, to save generated synthesizers and intermediate variables or not, default = `False`
  
Outputs (only when `--save` is set to `True`):
  * `models/retailer_embedding.model`, retailer embedding model
  * `models/TVAE_synthesizer.pkl`, synthesizer file
  * `models/retailer_map_grouped.pkl`, retailer mapping file to each sectors
  * `data/cc_data_processed.csv`, data file after pre-processing
  * `data/cc_data_input.csv`, data file after pre-processing and feature engineering, ready for training the synthesizer
  
Example:
```python
>> Python3 train.py --f 'data/cc_data.csv' --n 2000 --s True
```

### `generate.py`

Inputs:
  * `--y` or `--year`, integer, which year to forecast transactions, default is current year
  * `--m` or `--month`, integer, which month to forecast transactions, default is current month
  
Outputs:
  * `output/xxxx_xxx.csv`, generated transaction records, for instance, `output/2020_Dec.csv`
  
Example:
```Python
>> Python3 generate.py --y 2020 --m 12
```

## Data Pre-processing

I selected the following features for each transaction, removed missing or invalid values.

* Account Type
* Consumer Gender
* Consumer Age
* Transaction Date
* Normalized Retailer
* SIC Description (sector)
* Purchase Amount

## Feature Engineering

* One-hot-encoding
* Age
* Transaction Date
* Normalized Retailer

## Build Synthesizer

* How to train Tabular Variational Autoencoder (TVAE) on GCP

## Build Forecast Model

* API from Statistics Canada
* Forecast macroeconomic data using `prophet`
* Generalized linear model

## Future Work

* Deployment
