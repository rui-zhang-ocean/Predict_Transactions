import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sdgym.synthesizers import TVAESynthesizer
import datetime
import pickle
import random

df = pd.read_csv('/home/rzhang_dal/predict_transactions/cc_data.csv')
#df = df[:1000]

# remove unnecessary columns
col2remove = ['SIC Code', 'Return Amount', 'Reward Amount', 'Transaction ID', 
              'Account Identifier', 'Account Name', 'Account Number', 'Bank Name', 
              'Aggregator Name', 'Consumer ID', 'Consumer Created Date',
              'Transaction String', 'Posted Date', 'Data Creation Date', 
              'Consumer Postal Code', 'Consumer City Name','Ethnicity']
df.drop(col2remove, axis = 1, inplace = True, errors='ignore') 

# Only keep `purchase` rows for `Transaction Type`, and then remove `Trsansaction Type`
if 'Transaction Type' in df.columns:
    df = df[df['Transaction Type'] == 'purchase']
    df.drop('Transaction Type', axis = 1, inplace = True)
    
# calculate consumer age, any birth year after 2020 is converted to null, and then remove `Consumer Birth Year` column
if 'Consumer Birth Year' in df.columns:
    df['Age'] = df['Consumer Birth Year'].apply(lambda x: 2020 - int(x) if int(x) < 2020 else None)
    df.drop('Consumer Birth Year', axis = 1, inplace = True)
    
# convert `N\A` in `Transation date` into null
df['Transaction Date'].replace({"N\A":None}, inplace=True)
# convert `both` in `Consumer Gender` into null, only keep male and female
df['Consumer Gender'].replace({'both':None}, inplace=True)
# convert `investment_account` and `loans` in `Account Type` into null, only keep bank_account and credit_card
df['Account Type'].replace({'investment_account':None,'loans':None},inplace=True)
# remove missing values above
df.dropna(inplace = True)

# convert ‘Transaction Date’ into day_of_week (Mon/Tue.) and period_of_month (start, mid and end).
if 'Transaction Date' in df.columns:
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
    df['day_of_week'] = df['Transaction Date'].dt.day_name()
    df['day_of_month'] = df['Transaction Date'].dt.day
    df['period_of_month'] = df.apply(lambda x: 'start' if x.day_of_month <= 10 else 'mid' if x.day_of_month <=20 else 'end', axis = 1)
    df.drop(['Transaction Date','day_of_month'], axis = 1, inplace = True)
    
# 'SIC Description' (114) - only keep top N and group the rest into `other`
N = 9
def viewSICCounts(df,col_name):
    df_pivot = df.groupby(by = col_name).size().reset_index(name='Counts')
    df_pivot['Per (%)'] = (df_pivot['Counts'])/df.shape[0]*100
    df_pivot.sort_values(by = 'Counts',ascending = False,inplace = True)
    return df_pivot
    
df_pivot = viewSICCounts(df,'SIC Description')
list2keep = list(df_pivot.nlargest(N, 'Counts')['SIC Description'])
df['SIC Description'] = df['SIC Description'].apply(lambda x: x if x in list2keep else 'Other')

# 'Normalized Retailer' (2449) - 20 dimensions embedding
model = Word2Vec.load('/home/rzhang_dal/predict_transactions/perSICperPerson.model')

# remove records with minority retailers (the dictionary only keep retailer that appears at least 5 times)
df = df[df['Normalized Retailer'].isin(list(model.wv.vocab))]
retailerVec = model.wv[df['Normalized Retailer']]

# convert retailer vector array into dataframe
df_retailerVec = pd.DataFrame(retailerVec, columns=["retailerVec_%02d" % x for x in range(1,21)]) 

# one hot encoding for categorical columns except `Normalized Retailer`
df_dummy = df.copy()
df_dummy.drop('Normalized Retailer', axis = 1, inplace = True, errors='ignore')
df_dummy = pd.get_dummies(df_dummy)

# concatenate df_dummy and df_retailerVec
df_dummy.reset_index(inplace=True,drop=True)
df_retailerVec.reset_index(inplace=True,drop=True)
df_input = pd.concat([df_dummy, df_retailerVec], axis = 1, sort = False, ignore_index = False)

# randomly sample less than 3 million records
df_input = df_input.sample(n = 1000000)

# convert pd frame to np array and indicate categorical and oridinal columns
data = df_input.to_numpy()
categorical_columns = [x for x in range(2,46)]
ordinal_columns = [1]

# train the synthesizer
start = datetime.datetime.now()

print("TVAE starts training at ", start)

synthesizer = TVAESynthesizer()
synthesizer.fit(data, categorical_columns, ordinal_columns)

print("TVAE training time: " + str(datetime.datetime.now()-start))

# save the synthesizer
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # overwrite any existing file
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

save_object(synthesizer, '/home/rzhang_dal/predict_transactions/TVAE_synthesizer.pkl')

# check out sample
sampled = synthesizer.sample(1)
np.set_printoptions(suppress = True, precision = 2)
print(sampled)
