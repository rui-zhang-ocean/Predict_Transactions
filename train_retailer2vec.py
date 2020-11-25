import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import datetime


#---------------------------------INPUTS--------------------------------------
raw_data_file = 'data/cc_data.csv'

#----------------------------------FUNCTIONS----------------------------------
# create reatiler sets grouped by SIC and consumer
def groupBySICAndPerson(df):
    df_group = df.groupby(['SIC Description','Consumer ID']).agg(lambda x: x.tolist())
    training_data = df_group['Normalized Retailer'].tolist()
    training_data = [x for x in training_data if len(x) > 1]
    return training_data

def view_column_counts(df, col_name):
    df_pivot = df.groupby(by = col_name).size().reset_index(name='Counts')
    df_pivot['Per (%)'] = (df_pivot['Counts'])/df.shape[0]*100
    df_pivot.sort_values(by = 'Counts',ascending = False,inplace = True)
    return df_pivot

#----------------------------------MAIN---------------------------------------
# load data
print('Loading data.......')
df = pd.read_csv(raw_data_file)

# keep columns useful only
df = df[['Consumer ID','Normalized Retailer','SIC Description']]

# create training sets 
print('Creating training sets......')
start = datetime.datetime.now()
training_data = groupBySICAndPerson(df)
print("Sets creating time: " + str(datetime.datetime.now()-start))

# training retailer embedding
start = datetime.datetime.now()
print('Training retailer embedding......')
model = Word2Vec(sentences = training_data, # list of sets of retailers
                 iter = 10, # epoch
                 min_count = 5, # a retailer has to appear more than min_count times to be kept
                 size = 10, # hidden layer dimensions
                 workers = 4, # specify the number of threads to be used for training
                 sg = 1, # Defines the training algorithm. We will use skip-gram so 1 is chosen.
                 hs = 0, # Set to 0, as we are applying negative sampling.
                 negative = 10, # If > 0, negative sampling will be used. We will use a value of 5.
                 window = 9999999)
print("Model training time: " + str(datetime.datetime.now()-start))

# save the model
#model.save("models/retailer_embedding.model")
