import pandas as pd
import numpy as np

import sklearn.metrics as metric
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures


#################################################################################################

def baseline_models(y_train, y_validate):
    '''
    Takes in y_train and y_validate and returns a df of 
    baseline_mean and baseline_median and how they perform
    '''
    train_predictions = pd.DataFrame(y_train)
    validate_predictions = pd.DataFrame(y_validate)
    
    y_pred_mean = y_train.mean()
    train_predictions['y_pred_mean'] = y_pred_mean
    validate_predictions['y_pred_mean'] = y_pred_mean
    
    y_pred_median = y_train.median()
    train_predictions['y_pred_median'] = y_pred_median
    validate_predictions['y_pred_median'] = y_pred_median

    # create the metric_df as a blank dataframe
    metric_df = pd.DataFrame(data=[
    {
        'model': 'mean_baseline', 
        'RMSE_train': metric.mean_squared_error(
            y_train,
            train_predictions['y_pred_mean']) ** .5,
        'RMSE_validate': metric.mean_squared_error(
            y_validate,
            validate_predictions['y_pred_mean']) ** .5,
        'Delta': (( metric.mean_squared_error(
            y_train,
            train_predictions['y_pred_mean']) ** .5)-(metric.mean_squared_error(
            y_validate,
            validate_predictions['y_pred_mean']) ** .5))
    }])

    return metric_df.append(
            {
                'model': 'median_baseline', 
                'RMSE_train': metric.mean_squared_error(
                    y_train,
                    train_predictions['y_pred_median']) ** .5,
                'RMSE_validate': metric.mean_squared_error(
                    y_validate,
                    validate_predictions['y_pred_median']) ** .5,
                'Delta': (( metric.mean_squared_error(
                    y_train,
                    train_predictions['y_pred_median']) ** .5)-(metric.mean_squared_error(
                    y_validate,
                    validate_predictions['y_pred_median']) ** .5))
            }, ignore_index=True)


def get_lars(X_train, y_train, X_validate, y_validate):
    
    '''
    takes in x, y train, x, y, validate, returns rmse score for lasso lars model
    '''

    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

    # create the model object
    lars = LassoLars(fit_intercept=False)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lars.fit(X_train, y_train.home_value)

    # predict train
    y_train['value_pred_lars'] = lars.predict(X_train)

    # evaluate: rmse
    rmse_train = metric.mean_squared_error(y_train.home_value, y_train.value_pred_lars)**(1/2)

    # predict validate
    y_validate['value_pred_lars'] = lars.predict(X_validate)
    
    # evaluate: rmse
    rmse_validate = metric.mean_squared_error(y_validate.home_value, y_validate.value_pred_lars)**(1/2)

    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)
    print('===================================')

def get_tweedie(X_train, y_train, X_validate, y_validate):
    '''
    takes in x, y train, x, y, validate, returns rmse score for tweedie regressor model
    '''
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    
    # create the model object
    glm = TweedieRegressor(power=2, alpha=1)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(X_train, y_train.home_value)

    # predict train
    y_train['value_pred_glm'] = glm.predict(X_train)

    # evaluate: rmse
    rmse_train = metric.mean_squared_error(y_train.home_value, y_train.value_pred_glm)**(1/2)

    # predict validate
    y_validate['value_pred_glm'] = glm.predict(X_validate)

    # evaluate: rmse
    rmse_validate = metric.mean_squared_error(y_validate.home_value, y_validate.value_pred_glm)**(1/2)

    print("RMSE for GLM using Tweedie, power=2 & alpha=1\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)
    print('===================================')

    
def get_linreg(X_train, y_train, X_validate, y_validate):
    '''
    takes in x, y train, x, y, validate, returns rmse score for ols model
    '''
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    
    # create the model object
    lm = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(X_train, y_train.home_value)

    # predict train
    y_train['value_pred_lm'] = lm.predict(X_train)

    # evaluate: rmse
    rmse_train = metric.mean_squared_error(y_train.home_value, y_train.value_pred_lm)**(1/2)

    # predict validate
    y_validate['value_pred_lm'] = lm.predict(X_validate)

    # evaluate: rmse
    rmse_validate = metric.mean_squared_error(y_validate.home_value, y_validate.value_pred_lm)**(1/2)

    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)
    
    
    
def get_test(X_train, y_train, X_test, y_test):
    '''
    takes in x, y test, returns rmse score for ols model
    ''' 
    # We need y_train & y_test to be dataframes to append the new columns with predicted values.
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    # create the model object
    lm = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(X_train, y_train.home_value)

    # predict test
    y_test['value_pred_lm'] = lm.predict(X_test)

    # evaluate: rmse
    rmse_test = metric.mean_squared_error(y_test.home_value, y_test.value_pred_lm)**(1/2)

    print("RMSE for OLS using LinearRegression\nTest-Sample: ", rmse_test)