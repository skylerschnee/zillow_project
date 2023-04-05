#!/usr/bin/env python
# coding: utf-8

# In[67]:


#################################### imports #####################################

import pandas as pd
import os
import numpy as np
import env
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

############################# acquire main function ##############################


def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def acquire_zillow_data():
    '''
    wrangle_zillow_data will acuire the telco_churn data and 
    proceed to drop redundant columns and non-usefull info in 
    addition to encoding categorical variables
    '''
    filename = "zillow_data.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        query = '''
        select bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
        from properties_2017
        where (propertylandusetypeid like '261' OR propertylandusetypeid like '279')
        '''
        df = pd.read_sql(query, get_connection('zillow'))
        
        # saving to csv
        df.to_csv('zillow_data.csv', index = False)

        return pd.read_csv(filename)
    
################################ clean Data #####################################     


def clean_zillow_data(df):
    '''
    clean_zillow_data will take in a single pandas dataframe, 
    with the expected formatting observed in zillow df.
    It will drop nulls,
    It will convert most dtypes to int except tax_val_usd,,
    And it will rename columns for easier reading 
    
    Arguments: df. a pandas dataframe
    return: df, a pandas dataframe (cleaned)
    '''
    #drop nulls
    df = df.dropna()
    #rename columns for easier reading
    df.rename(columns={'bedroomcnt':'bedrooms','bathroomcnt':'bathrooms', 
                       'calculatedfinishedsquarefeet':'sqft','taxvaluedollarcnt':'home_value',
                       'taxamount':'yearly_tax', 'yearbuilt':'year_built', 'fips': 'county'}, inplace = True)
   
    # converting column datatypes
    df.county = df.county.astype(object)
    df.year_built = df.year_built.astype(object)
    
    # Create a dictionary mapping the old values to their corresponding names
    value_map = {'6037.0': 'los_angeles', '6059.0': 'orange', '6111.0': 'ventura'}

    # Use replace to rename values in the county column
    df['county'] = df['county'].replace({6037.0: 'los_angeles', 6059.0: 'orange', 6111.0: 'ventura'})
    
    df.reset_index(drop=True)
    return df

################################ Split Data #####################################                         
                         
def split_zillow(df):
    '''
    split_zillow will take in a single pandas df referencing a cleaned
    version of zillow data, and will then split the data into train,
    validate, and test sets
    
    Arguments: df. a pandas dataframe
    return: train, validate, test: the pandas df split from orginal df 
    '''
    train_val, test = train_test_split(df, random_state = 828, train_size = 0.8)
    train, validate = train_test_split(train_val, random_state = 828, train_size = 0.7)
    return df, train, validate, test

################################ full wrangle Data ##################################### 

def wrangle_zillow():
    '''
    wrangle_zillow will go through the process of acquiring zillow data from 
    a local .csv if present, if not, aquire through a sql query, save the data to a local .csv
    then proceed with cleaning the data, then splitting into train, test, and validate
    '''
    return split_zillow(
        clean_zillow_data(
            acquire_zillow_data()))

################################ scale Data ##################################### 

def scale_data(train, 
               validate, 
               test, 
               columns_to_scale=['bedrooms', 'bathrooms', 'sqft', 'yearly_tax'],
               return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    # make copies of our original data so we dont gronk up anything
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #     make the thing
    scaler = sklearn.preprocessing.MinMaxScaler()
    #     fit the thing
    scaler.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled


# In[ ]:




