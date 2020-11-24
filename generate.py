import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import pickle
import torch
import io
import random
from gensim.models import Word2Vec

#---------------------------------INPUTS--------------------------------------
# define which synthesier to use
synthesizer_file = 'TVAE_synthesizer.pkl'
# how many samples to generate
N = 1000
# which year and month to generate
Y = 2020; M = 12


#----------------------------------FUNCTIONS----------------------------------
# function to work around loading a GPU generated pickle on local CPU machine
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

# return series of dummy variables with given column name
def reverse_dummy(df, col_name):
    # get index of columns that starts with col_name, for example, Gender_male, Gender_female for col_name = `Gender`
    idx = [i for i, s in enumerate(list(df.columns)) if col_name in s]
    tmp = df.iloc[:,idx]
    # find column name for absolute max in each row and put into one series
    df_output = pd.Series(tmp.abs().idxmax(axis=1), name = col_name)
    
    # remove strings with col_name plus underscore
    df_output = df_output.map(lambda x: x.replace(col_name + '_',''))
    return df_output

# reverse age from age range
def return_age(age_range):
    age_range = age_range.replace('(','').replace(']','').replace(' ','')
    r1 = int(age_range.split(',')[0])
    r2 = int(age_range.split(',')[1])
    return random.choice(list(range(r1+1, r2+1)))

# find out retailer in the same SIC and also with the most similar vector
def return_retailer_SIC(model, retailer_map_grouped, sector, vector):
    if pd.isnull(sector):
        return None
    else:
        word_list = retailer_map_grouped[sector]
        min_idx = model.wv.distances(vector, other_words = word_list).argmin()
        return word_list[min_idx]
    
    

#----------------------------------MAIN---------------------------------------
# load synthesizer from saved object
with open('models/' + synthesizer_file, 'rb') as input:
    synthesizer = CPU_Unpickler(input).load()
synthesizer.device = 'cpu'

# generate samples
sample = synthesizer.sample(N)

# load column names for synthesized data
df_input = pd.read_csv('data/cc_data_input_10emb_ageCat70.csv')
input_columns = list(df_input.columns)[1:]
df_sample = pd.DataFrame(sample, columns=input_columns) 

# reverse each categorical variable to readable ones
df_purchase = df_sample.iloc[:,[0]]
df_age = reverse_dummy(df_sample,'Age Range')
df_account = reverse_dummy(df_sample,'Account Type')
df_gender = reverse_dummy(df_sample,'Consumer Gender')
df_SIC = reverse_dummy(df_sample,'SIC Description')
df_dw = reverse_dummy(df_sample,'day_of_week')
df_pm = reverse_dummy(df_sample,'period_of_month')

df_reverse = pd.concat([df_purchase,df_age,df_account,df_gender,df_SIC,df_dw,df_pm], axis=1)
df_reverse = df_reverse[(df_reverse['Purchase Amount'] > 0)]
df_reverse.reset_index(drop = True, inplace = True)

# return age from age range
df_reverse['Age'] = df_reverse.apply(lambda x: return_age(x['Age Range']), axis = 1)







        
