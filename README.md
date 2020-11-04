# Project Summary

* Building a predictive model to continuously forecast customer behaviour and optimize marketing strategies with historical transaction data.
* Implementing tabular variational autoencoders to synthesize transaction records, with item embedding over 2000+ retailers.
* Applying regression models using macroeconomic data to constrain monthly transaction counts, where data is collected with an automated web scraper.
* Collaborating with head data scientist at Arima to deploy the model on GCP.

## 0_data_description

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
* ...

## 1_data_cleaning_and_feature_engineer

### Missing values

### Age

### Transaction Date

### Retailer embedding using Item2Vec


## 2_build_synthesizer

Tabular Variational Autoencoder (TVAE)

## 3_build_forecast_model

### Web Scraper from Statistic Canada

### Forecast future macroeconomic data

### Regression Model

## 4_deployment

## 5_future_work
