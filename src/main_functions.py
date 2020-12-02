import pickle
import torch
import io
import pandas as pd
import random
from fbprophet import Prophet
import sys


# create reatiler sets grouped by SIC and consumer
def groupBySICAndPerson(df):
    df_group = df.groupby(['SIC Description','Consumer ID']).agg(lambda x: x.tolist())
    training_data = df_group['Normalized Retailer'].tolist()
    training_data = [x for x in training_data if len(x) > 1]
    return training_data

# view counts and percentage for each column's elements
def view_column_counts(df, col_name):
    df_pivot = df.groupby(by = col_name).size().reset_index(name='Counts')
    df_pivot['Per (%)'] = (df_pivot['Counts'])/df.shape[0]*100
    df_pivot.sort_values(by = 'Counts',ascending = False,inplace = True)
    return df_pivot

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # overwrite any existing file
        pickle.dump(obj, output, pickle.DEFAULT_PROTOCOL)
        
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
    
def return_day_index(period_of_month, year, month):
    if period_of_month == 'start':
        return list(range(1,11))
    elif period_of_month == 'mid':
        return list(range(11,21))
    else:
        return list(range(21, pd.Period(str(year) + '-' + str(month)).days_in_month + 1))
    
# function to convert day_of_week and period_of_month back to Date, given a year and a month
# Example:
# input: day_of_week = 'Monday'; period_of_month = 'start'; Y = 2020; M = 2
# output: 2020-02-03 00:00:00
def return_date(day_of_week, period_of_month, Y, M):
    # return list of days
    D = return_day_index(period_of_month, Y, M)

    tmp = pd.DataFrame({'year': [str(Y) for i in range(len(D))],
                        'month': [str(M) for i in range(len(D))],
                        'day': D})
    
    # create table with each row of year, month and day in given period
    date_period = pd.to_datetime(tmp[['year', 'month', 'day']])

    # locate index of which date is the given day_of_week
    idx_list = [i for i, s in enumerate(date_period.dt.strftime('%A')) if day_of_week in s]

    # randomly pick up one index, since it's possible one period has multiple given weekday (say Monday)
    idx = random.choice(idx_list)

    return date_period[idx]
    
def forecast_eco(df_eco, eco_var, Y = pd.to_datetime('now').year, M = pd.to_datetime('now').month, plot = False):
    df_input = df_eco[eco_var].reset_index().rename(columns = {"Date": "ds", eco_var : "y"})
    df_input.dropna(inplace = True)
    
    m = Prophet();
    m.fit(df_input);
    
    # `freq` indicates units, here we use start of the month (MS), to predict extra month(s) in future
    # extra months number is determined by the difference between current month and the latest data available month
    mon_diff = M - df_input['ds'].iloc[-1].month + 12 * (Y - df_input['ds'].iloc[-1].year)
    if mon_diff <= 0:
        sys.exit('Forecast year and month must be current or future date, please specify a different date.')
    else:
        future = m.make_future_dataframe(periods = mon_diff, freq = "MS")    
    forecast = m.predict(future)
    
    if plot == True:
        fig = m.plot(forecast, figsize=(8,3))

    return forecast