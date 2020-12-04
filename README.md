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

I selected the following features for each transaction record, and removed missing or invalid values.

* Account Type
* Consumer Gender
* Consumer Age
* Transaction Date
* Normalized Retailer
* SIC Description (sector)
* Purchase Amount

## Feature Engineering

Categorical variables with number of values smaller than 10 will be treated with one-hot-encoding, including `Account Type`, `Consumer Gender` and `SIC Description`. Note that `SIC Description` only kept top 9 values and grouped the rest into `other`.

`Age` was treated as continuous numerical variable at first but the synthesized age distribution didn't look optimal. I fixed it by converting age into age range (i.e. 20-25, 25-30, 30-35, etc).

`Transaction Date` was broken down into two categorical values `period of month` and `day of week`. `period of month` contains variables `start` (1-10 days), `mid` (11-20 days) and `end` (21 to the end of the month days). `day of week` refers to individual day of the week (Mon, Tue, etc). Thus, the synthesizer can reflect daily and weekly variability. 

`Normalized Retailer` is the most tricky variable to deal with. It contains over 2000 variables so embedding is needed to reduce the dimensions. I applied `item2vec` to convert it into 10 dimensions embeddings. I tested with different schemes of selecting training sets, the best scheme is to select training sets based on the same sector and customer.

![retailer embedding](https://github.com/rui-zhang-ocean/Predict_Transactions/blob/master/experiments/figs/retailer2vec/exp_07_10emb.png "retailer embedding")

## Build Synthesizer

### Tabular Variational Autoencoder (TVAE)

I used Tabular Variational Autoencoder (TVAE) from [sdgym](https://github.com/sdv-dev/SDGym/blob/master/sdgym/synthesizers/tvae.py), which requires specification of which columns are ordinal, categorical or numerical values. The training could take up to a few days time depending on the number of the records being used. GPU is recommended.

A post-correction is applied to `Transaction Amount` for each retailer by scaling with the average transaction amounts for each retailer from input data.

### Evaluation of the synthesizer

| Variables        | Input data      | Synthesized data  |
| ---------------- |:---------------:|:-----------------:|
| Age                   | ![age histogram](https://github.com/rui-zhang-ocean/Predict_Transactions/blob/master/experiments/figs/eda/age_hist_input.png "age histogram input") | ![age histogram](https://github.com/rui-zhang-ocean/Predict_Transactions/blob/master/experiments/figs/eda/age_hist_syn_ageCat70.png "age histogram synthesized") |
| Total Purchase Amount       | ![purchase histogram](https://github.com/rui-zhang-ocean/Predict_Transactions/blob/master/experiments/figs/eda/purchase_hist_input.png "purchase histogram input")       | ![purchase histogram](https://github.com/rui-zhang-ocean/Predict_Transactions/blob/master/experiments/figs/eda/purchase_hist_syn_ageCat70_corr.png "purchase histogram synthesized") |
| Purchase amount per sector   | ![purchase sector](https://github.com/rui-zhang-ocean/Predict_Transactions/blob/master/experiments/figs/eda/purchase_SIC%20Description_input.png "purchase sector input")| ![purchase sector](https://github.com/rui-zhang-ocean/Predict_Transactions/blob/master/experiments/figs/eda/purchase_SIC%20Description_synthesized.png "purchase sector synthesized")|
| Purchase amount per retailer | ![purchase retailer](https://github.com/rui-zhang-ocean/Predict_Transactions/blob/master/experiments/figs/eda/purchase_Normalized%20Retailer_input.png "purchase retailer input") |   ![purchase retailer](https://github.com/rui-zhang-ocean/Predict_Transactions/blob/master/experiments/figs/eda/purchase_Normalized%20Retailer_synthesized.png "purchase retailer synthesized") |

## Build Forecast Model

* API from Statistics Canada

[stats_can](https://stats-can.readthedocs.io/en/latest/), a python library that wraps up the API from Statistics Canada

* Forecast macroeconomic data using `prophet`
* Generalized linear model

## Future Work

* Deployment
