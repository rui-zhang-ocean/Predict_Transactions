# Project Summary

* Building a predictive model to continuously forecast customer behaviour and optimize marketing strategies with historical transaction data.
* Implementing tabular variational autoencoders to synthesize transaction records, with item embedding over 2000+ retailers.
* Applying regression models using macroeconomic data to constrain monthly transaction counts, where data is collected with API from Statistics Canada.

## How to use the package

The package includes two scripts. `train.py` uses raw data to train synthesizers. `generate.py` uses synthesizers (trained by `train.py`) to generate transaction records with specified year and month.

### `train.py`

Inputs: 
  * `--f` or `--filename`, default = `data/cc_data.csv`, string, raw data file directory
  * `--n` or `--samplenum`, default = `1000`, integer, number of samples randomly chosen for training
  * `--s` or `--save`, default = `False`, boolean, to save generated synthesizers and intermediate variables or not
  
Outputs (only when `--save` is set to True):
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
  * `--y` or `--year`, integer, default is current year
  * `--m` or `--month`, integer, default is current month
  
Outputs:
  * `output/xxxx_xxx.csv`, generated transaction records, for instance, `output/2020_Dec.csv`
  
Example:
```Python
>> Python3 generate.py --y 2020 --m 12
```

## Data Description

For each transaction, the table contains information below:

* Account Type
* Consumer Gender
* Consumer Age
* Consumer Postal Code
* Consumer Birth Year
* Transaction Type
* Transaction Date
* Normalized Retailer
* SIC Description (sector)
* Purchase Amount

## Data Cleaning and Feature Engineering

* Missing values
* Age
* Transaction Date
* Retailer embedding using Item2Vec

## Build Synthesizer

* Tabular Variational Autoencoder (TVAE)
* Compute engine setup on Google Cloud

## Build Forecast Model

* Web Scraper for Statistic Canada
* Forecast macroeconomic data using `prophet`
* Regression Model

## Deployment

## Future Work
