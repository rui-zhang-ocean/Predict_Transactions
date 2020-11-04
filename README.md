# Project Summary

* Building a predictive model to continuously forecast customer behaviour and optimize marketing strategies with historical transaction data.
* Implementing tabular variational autoencoders to synthesize transaction records, with item embedding over 2000+ retailers.
* Applying regression models using macroeconomic data to constrain monthly transaction counts, where data is collected with an automated web scraper.
* Collaborating with head data scientist at Arima to deploy the model on GCP.

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
