#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import ignore warninings
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

#----------------------------------------------------------------#

def plot_residuals(y, yhat):
    '''
    This function takes in actual value and predicted value 
    then creates a scatter plot of those values
    '''
    residuals = y - yhat
    
    plt.scatter(x=y, y=residuals)
    plt.xlabel('Home Value')
    plt.ylabel('Residuals')
    plt.title('Residual vs Home Value Plot')
    plt.show()
    

def regression_errors(y, yhat):
    '''
    This function takes in actual value and predicted value 
    then outputs: the sse, ess, tss, mse, and rmse
    '''
    MSE = mean_squared_error(y, yhat)
    SSE = MSE * len(y)
    RMSE = math.sqrt(MSE)
    ESS = ((yhat - y.mean())**2).sum()
    TSS = ESS + SSE
           
    print(f'SSE: {SSE: .4f}')
    print(f'ESS: {ESS: .4f}')
    print(f'TSS: {TSS: .4f}')
    print(f'MSE: {MSE: .4f}')
    print(f'RMSE: {RMSE: .4f}')

    return MSE, SSE, RMSE, ESS, TSS


def baseline_mean_errors(y):
    baseline = np.repeat(y.mean(), len(y))
    
    MSE = mean_squared_error(y, baseline)
    SSE = MSE * len(y)
    RMSE = MSE**.5
    
    print(f'sse_baseline: {SSE: .4f}')
    print(f'mse_baseline: {MSE: .4f}')
    print(f'rmse_baseline: {RMSE: .4f}')
    
    return MSE, SSE, RMSE


def better_than_baseline(y, yhat):
    '''
    This function takes in the target and the prediction
    then returns a print statement 
    to inform us if the model outperforms the baseline
    '''
    SSE, ESS, TSS, MSE, RMSE = regression_errors(y, yhat)
    
    SSE_baseline, MSE_baseline, RMSE_baseline = baseline_mean_errors(y)
    
    if SSE < SSE_baseline:
        print('My OSL model performs better than baseline')
    else:
        print('My OSL model performs worse than baseline. :( )')


# In[ ]:




