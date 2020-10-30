import pandas as pd
import numpy as np

df_input = pd.read_csv('data/cc_data_input.csv', index_col = False)

# remove second column
df_input = df_input.drop(['Unnamed: 0'], axis=1)

df_input['day_of_week_Tuesday'].value_counts()

df_input['period_of_month_end'].value_counts()
