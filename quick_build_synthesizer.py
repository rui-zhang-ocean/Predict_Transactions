import pandas as pd
import numpy as np
from sdgym.synthesizers import TVAESynthesizer
import datetime
import pickle
import random

df_input = pd.read_csv('cc_data_input_10emb_ageCat.csv', index_col = False)

# remove second column
df_input = df_input.drop(['Unnamed: 0'], axis=1)

# randomly sample less than 3 million records
df_input = df_input.sample(n = 100000, random_state = 1)

# convert pd frame to np array and indicate categorical and oridinal columns
data = df_input.to_numpy()
#categorical_columns = [x for x in range(2,26)]
#ordinal_columns = [1]

# train the synthesizer
start = datetime.datetime.now()

print("TVAE starts training at ", start)

synthesizer = TVAESynthesizer()
synthesizer.fit(data)
#synthesizer.fit(data, categorical_columns, ordinal_columns)

print("TVAE training time: " + str(datetime.datetime.now()-start))

# save the synthesizer
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # overwrite any existing file
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

save_object(synthesizer, '/home/rzhang_dal/project/TVAE_synthesizer_10emb_noCat_ageCat_0.1M.pkl')

# check out sample
sampled = synthesizer.sample(1)
np.set_printoptions(suppress = True, precision = 2)
print(sampled)
