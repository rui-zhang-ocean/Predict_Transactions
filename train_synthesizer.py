import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sdgym.synthesizers import TVAESynthesizer
import datetime
import pickle
from collections import defaultdict


#---------------------------------INPUTS--------------------------------------
raw_data_file = 'data/cc_data.csv'
retailer_embedding_file = 'models/retailer_embedding.model'
sample_num = 1000 # how many records are used for training

#----------------------------------FUNCTIONS----------------------------------
# view counts and percentage for each column's elements
def view_column_counts(df, col_name):
    df_pivot = df.groupby(by = col_name).size().reset_index(name='Counts')
    df_pivot['Per (%)'] = (df_pivot['Counts'])/df.shape[0]*100
    df_pivot.sort_values(by = 'Counts',ascending = False,inplace = True)
    return df_pivot

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # overwrite any existing file
        pickle.dump(obj, output, pickle.DEFAULT_PROTOCOL)

#----------------------------------MAIN---------------------------------------
# load data
print('Loading raw data......')
df = pd.read_csv(raw_data_file)

# remove unnecessary columns
print('Pre-processing......')
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

# remove missing values
df.dropna(inplace = True)


# create retailer mapping file
retailer_list = list(df['Normalized Retailer'].unique())
SIC_list = list(df['SIC Description'].unique())

# create dictionary from SIC to retailer
print('Generating retailer SIC mapping file......')
retailer_map = defaultdict(list) # DO NOT USE dict.fromkeys, which appends retailer to every key

for i in range(len(retailer_list)):
    tmp_SIC = df.loc[df['Normalized Retailer'] == retailer_list[i]]['SIC Description'].unique()[0] 
    retailer_map[tmp_SIC].append(retailer_list[i])

model = Word2Vec.load(retailer_embedding_file)
# only keep values in the model (more than 5 times appearance)
for key, value in retailer_map.items():
    retailer_map[key] = list(set(value) & set(model.wv.vocab))
    
# group other SIC (after top N) into other
df_pivot = view_column_counts(df,'SIC Description')
list2keep = list(df_pivot.nlargest(9, 'Counts')['SIC Description'])
other_list = []    
for key, value in retailer_map.items():
    if key not in list2keep:
        other_list = other_list + value

# delete other group key-values in retailer_map
retailer_map_grouped = {k: retailer_map[k] for k in list2keep}

# add other key-value pairs
retailer_map_grouped['Other'] = other_list

# save the grouped retailer_map
#f = open("models/retailer_map_grouped.pkl","wb")
#pickle.dump(retailer_map_grouped,f)
#f.close()

# save processed file
#df_processed = df.copy()
#df_processed.to_csv('data/cc_data_processed.csv')

print('Feature engineering.......')
# convert ‘Transaction Date’ into day_of_week (Mon/Tue.) and period_of_month (start, mid and end).
if 'Transaction Date' in df.columns:
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
    df['day_of_week'] = df['Transaction Date'].dt.day_name()
    df['day_of_month'] = df['Transaction Date'].dt.day
    df['period_of_month'] = df.apply(lambda x: 'start' if x.day_of_month <= 10 else 'mid' if x.day_of_month <=20 else 'end', axis = 1)
    df.drop(['Transaction Date','day_of_month'], axis = 1, inplace = True)
    
# bucket age into categorical values every 5 years
bin = list(range(20,80,5))
age_cat = pd.cut(df.Age, bin).to_frame()
age_cat.columns = ['Age Range'] 

if "Age Range" not in df:
    df = pd.concat([df, age_cat],axis = 1)    
df.drop(['Age'], axis = 1, inplace = True, errors = 'ignore')
df.dropna(inplace = True)

# keep top 9 SIC categories and group the rest to `other`
df_pivot = view_column_counts(df,'SIC Description')
list2keep = list(df_pivot.nlargest(9, 'Counts')['SIC Description'])
print("SIC to keep: ", list2keep)
df['SIC Description'] = df['SIC Description'].apply(lambda x: x if x in list2keep else 'Other')

# retailer to embeddings
df = df[df['Normalized Retailer'].isin(list(model.wv.vocab))]
retailerVec = model.wv[df['Normalized Retailer']]

# convert retailer vector array into dataframe
df_retailerVec = pd.DataFrame(retailerVec, columns=["retailerVec_%02d" % x for x in range(1,(retailerVec.shape[1])+1)]) 

# one hot encoding for categorical columns except `Normalized Retailer`
df_dummy = df.copy()
df_dummy.drop('Normalized Retailer', axis = 1, inplace = True, errors='ignore')
df_dummy = pd.get_dummies(df_dummy)

# concatenate df_dummy and df_retailerVec
df_dummy.reset_index(inplace=True,drop=True)
df_retailerVec.reset_index(inplace=True,drop=True)
df_input = pd.concat([df_dummy, df_retailerVec], axis = 1, sort = False, ignore_index = False)

# save input file
#df_input.to_csv('data/cc_data_input.csv')

# train the synthesizer
df_input_sample = df_input.sample(n = sample_num)
data = df_input_sample.to_numpy()

start = datetime.datetime.now()
print("TVAE starts training at ", start)
synthesizer = TVAESynthesizer()
synthesizer.fit(data)
print("TVAE training time: " + str(datetime.datetime.now()-start))

# save the synthesizer
#save_object(synthesizer, 'models/TVAE_synthesizer_test.pkl')
        
# check out sample
sampled = synthesizer.sample(1)
np.set_printoptions(suppress = True, precision = 5)
print('Sample data from synthesizer......')
print(sampled)
