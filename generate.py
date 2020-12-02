import pandas as pd
import numpy as np
import pickle
import torch
import io
import random
from sklearn.linear_model import TweedieRegressor
from sklearn import preprocessing
import stats_can
from fbprophet import Prophet
from datetime import datetime
from gensim.models import Word2Vec
import sys
sys.path.append("./src")
from main_functions import *
import argparse


#---------------------------------INPUTS--------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--y', '--year', type = int, default = datetime.now().year,
                    help = 'which year to predict transactions')
parser.add_argument('--m', '--month', type = int, default = datetime.now().month,
                    help = 'which month to predict transactions')

args = parser.parse_args()
Y = args.y
M = args.m


#----------------------------------MAIN---------------------------------------
# define which files to use
synthesizer_file = 'models/TVAE_synthesizer.pkl'
retailer_map_file = 'models/retailer_map_grouped.pkl'
retailer_embedding_file = 'models/retailer_embedding.model'
data_processed_file = 'data/cc_data_processed.csv'
data_input_file = 'data/cc_data_input.csv'

# load processed input data to 
# 1, calculate purchase amount and transaction counts by month and
# 2, check mean purchase amount for each retailer (later)
print('Loading processed data file......')
df_processed = pd.read_csv(data_processed_file)
df_processed.index = pd.to_datetime(df_processed['Transaction Date']) # Use date as index
df_processed.drop('Transaction Date', axis = 1, inplace = True, errors='ignore')

# table with transaction counts and total amount aggregated for each month
df_group = df_processed.groupby([pd.Grouper(freq = 'M')]).agg({'Purchase Amount': {'count', 'sum'}}).rename(columns={'count':'Transaction_Count','sum':'Purchase_Sum'})
df_group.columns = df_group.columns.droplevel(0)

# only keep data from Sep 2018 to Jan 2020
df_period = df_group.loc['2018-09-01':'2020-01-31']

# load marco-economic data from statistics Canada
eco_vec_map = {'CPI':'v41690973',
               'Exchange_Rate_USD':'v111666275',
               'GDP':'v65201210',
               'Unemployment_Rate':'v91506256',
               'TSX':'v122620'}
vectors = list(eco_vec_map.values())
print('Scraping eco data from Stats Canada......')
df_eco = stats_can.sc.vectors_to_df(vectors, periods = 36) # get data in the last 36 months
inv_map = {v: k for k, v in eco_vec_map.items()}
df_eco.columns = df_eco.columns.to_series().map(inv_map)
df_eco.index.names = ['Date']

# Extract eco data from Sep 2018 to Jan 2020
df_eco_sel = df_eco.loc['2018-09-01':'2020-01-31']

# put together eco and transaction counts for regression
df_all = pd.concat([df_eco_sel, df_period.set_index(df_eco_sel.index)], axis = 1)
y_train = df_all['Transaction_Count'].values
X_train = df_all[['CPI','Exchange_Rate_USD','GDP','Unemployment_Rate','TSX']]

# generalized linear model
glm = TweedieRegressor(power = 1, alpha = 0.5, link='log') # Poisson distribution
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
glm.fit(X_train_scaled, y_train)

# predict eco data for given year and month
df_future = pd.DataFrame(columns=['Date'])
for i, eco_var in enumerate(list(eco_vec_map.keys())):
    print("Forecasting " + eco_var + ' ' + str(Y) + ' ' + datetime.strptime(str(M), "%m").strftime("%b"))
    tmp = forecast_eco(df_eco, eco_var, Y, M)
    tmp = tmp[['ds','trend']]
    tmp.rename(columns = {'ds':'Date','trend':eco_var}, inplace = True)
    df_future = df_future.merge(tmp, on = 'Date', how = 'right')

# predict transaction count using the glm model
eco_forecast = df_future.tail(1)[['CPI','Exchange_Rate_USD','GDP','Unemployment_Rate','TSX']]
transaction_count_forecast = glm.predict(scaler.transform(eco_forecast)).astype(int)[0]


# load synthesizer from saved object
with open(synthesizer_file, 'rb') as input:
    synthesizer = CPU_Unpickler(input).load()
synthesizer.device = 'cpu'

# generate 
print('Generating synthesized data with %i samples......' % transaction_count_forecast)
sample = synthesizer.sample(transaction_count_forecast)

# load column names for synthesized data
df_input = pd.read_csv(data_input_file)
input_columns = list(df_input.columns)[1:]
df_sample = pd.DataFrame(sample, columns=input_columns) 

# reverse each categorical variable to readable ones
print('Making synthesized data readable......')
df_purchase = df_sample.iloc[:,[0]]
df_age = reverse_dummy(df_sample,'Age Range')
df_account = reverse_dummy(df_sample,'Account Type')
df_gender = reverse_dummy(df_sample,'Consumer Gender')
df_SIC = reverse_dummy(df_sample,'SIC Description')
df_dw = reverse_dummy(df_sample,'day_of_week')
df_pm = reverse_dummy(df_sample,'period_of_month')

df_reverse = pd.concat([df_purchase,df_age,df_account,df_gender,df_SIC,df_dw,df_pm], axis=1)
df_reverse.reset_index(drop = True, inplace = True)

# return age from age range
df_reverse['Age'] = df_reverse.apply(lambda x: return_age(x['Age Range']), axis = 1)

# load the retailer_map
with open(retailer_map_file, 'rb') as input:
    retailer_map_grouped = pickle.load(input)  
    
# reverse retailerVec back to retailers
idx = [i for i, s in enumerate(list(df_sample.columns)) if 'retailerVec' in s]
df_retailerVec = df_sample.iloc[:,idx]  
df_retailerVec['retailerVec'] = df_retailerVec.values.tolist()
df_SIC_vector = pd.concat([df_reverse['SIC Description'], df_retailerVec['retailerVec']],axis = 1)

retailer2vec_model = Word2Vec.load(retailer_embedding_file)
df_reverse['Normalized Retailer'] = df_SIC_vector.apply(lambda x: return_retailer_SIC(retailer2vec_model, retailer_map_grouped, x['SIC Description'],x['retailerVec']), axis = 1)

# given period_of_month (start, mid or end) and year (1989) and month (6), return index of the possible days
df_reverse['Transaction Date'] = df_reverse.apply(lambda x: return_date(x.day_of_week, x.period_of_month, Y, M), axis = 1)

# remove negative purchase amount
df_reverse = df_reverse[(df_reverse['Purchase Amount'] > 0)]
df_reverse.reset_index(drop=True, inplace=True)

# create columns that are mean_byRetailer_input, mean_byRetailer_syn, and apply their ratio to adjust Purchase Amount
print('Applying post-correction to transaction amount for each retailer.......')
df_reverse['mean_byRetailer_syn'] = df_reverse['Normalized Retailer'].map(df_reverse.groupby(['Normalized Retailer'])['Purchase Amount'].mean())
df_reverse['mean_byRetailer_input'] = df_reverse['Normalized Retailer'].map(df_processed.groupby(['Normalized Retailer'])['Purchase Amount'].mean())
df_reverse['Purchase Amount Corrected'] = df_reverse.apply(lambda x: x['Purchase Amount'] * x.mean_byRetailer_input / x.mean_byRetailer_syn, axis = 1)

# drop intermediate values
col2drop = ['Age Range','day_of_week','period_of_month','mean_byRetailer_syn','mean_byRetailer_input']
df_reverse.drop(col2drop, axis = 1, inplace = True, errors = 'ignore')

# save synthesized data into csv file
output_filename = 'output/' + str(Y) + '_' + datetime.strptime(str(M), "%m").strftime("%b") + '.csv'
print('Saving file at ' + output_filename + '......')
df_reverse.to_csv(output_filename)
        