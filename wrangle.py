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

def remove_outliers(df,feature_list):
    ''' utilizes IQR to remove data which lies beyond 
    three standard deviations of the mean
    '''
    for feature in feature_list:
    
        #define interquartile range
        Q1= df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        #Set limits
        upper_limit = Q3 + 3 * IQR
        lower_limit = Q1 - 3 * IQR
        #remove outliers
        df = df[(df[feature] > lower_limit) & (df[feature] < upper_limit)]

    return df

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
    
    #remove outliers
    df = remove_outliers(df,['bedrooms','bathrooms','sqft','home_value'])
    
    df.reset_index(drop=True)
    return df

################################ Split Data #####################################                         
                         
def split_zillow(df):
    '''
    split_zillow will take in a single pandas df referencing a cleaned
    version of zillow data, and will then split the data into train,
    validate, and test sets stratifying on home_value
    
    Arguments: df. a pandas dataframe
    return: train, validate, test: the pandas df split from orginal df 
    '''
    train_val, test = train_test_split(df, random_state = 828, train_size = 0.8)
    train, validate = train_test_split(train_val, random_state = 828, train_size = 0.7)
    return train, validate, test




def wrangle_zillow():
    '''
    wrangle_zillow will go through the process of acquiring zillow data from 
    a local .csv if present, if not, aquire through a sql query, save the data to a local .csv
    then proceed with cleaning the data, then splitting into train, test, and validate
    '''
    return clean_zillow_data(acquire_zillow_data())

################################ scale and prep for modeling ##################################### 

def driver_sets(train, validate, test, features):
    train = train[features]
    validate=validate[features]
    test=test[features]
    return train, validate, test

def scale_zillow(train, validate, test):
    '''
    Takes in train, validate, test and scales those features.
    Returns df with new columns with scaled data
    '''
    scale_features=['bedrooms', 'bathrooms', 'sqft']
    
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    minmax = sklearn.preprocessing.MinMaxScaler()
    minmax.fit(train[scale_features])
    
    train_scaled[scale_features] = pd.DataFrame(minmax.transform(train[scale_features]),
                                                  columns=train[scale_features].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[scale_features] = pd.DataFrame(minmax.transform(validate[scale_features]),
                                               columns=validate[scale_features].columns.values).set_index([validate.index.values])
    
    test_scaled[scale_features] = pd.DataFrame(minmax.transform(test[scale_features]),
                                                 columns=test[scale_features].columns.values).set_index([test.index.values])
    
    return train_scaled, validate_scaled, test_scaled


def prep_for_model(train, validate, test, target):
    '''
    Takes in train, validate, and test data frames
    then splits  for X (all variables but target variable) 
    and y (only target variable) for each data frame
    '''
    #scale data
    train_scaled, validate_scaled, test_scaled = scale_zillow(train, validate, test)
    
    #make list of cat variables to make dummies for
    cat_vars = ['county']
    
    X_train = train_scaled
    X_train = X_train.drop(columns=['home_value'])
    dummy_df_train = pd.get_dummies(X_train[cat_vars], dummy_na=False, drop_first=[True, True])
    X_train = pd.concat([X_train, dummy_df_train], axis=1).drop(columns=cat_vars)
    y_train = train[target]

    X_validate = validate_scaled
    X_validate = X_validate.drop(columns=['home_value'])
    dummy_df_validate = pd.get_dummies(X_validate[cat_vars], dummy_na=False, drop_first=[True, True])
    X_validate = pd.concat([X_validate, dummy_df_validate], axis=1).drop(columns=cat_vars)
    y_validate = validate[target]

    X_test = test_scaled
    X_test = X_test.drop(columns=['home_value'])
    dummy_df_test = pd.get_dummies(X_test[cat_vars], dummy_na=False, drop_first=[True, True])
    X_test = pd.concat([X_test, dummy_df_test], axis=1).drop(columns=cat_vars)
    y_test = test[target]

    return X_train, y_train, X_validate, y_validate, X_test, y_test

# In[ ]:




