import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pearsonr, kruskal

#--------------------------Stat Tests------------------------

def get_kruskal_county(train):
    '''This function takes in train data and performs a Kruskal-Wallis test,
    comparing median home values in LA County with those in Orange and Ventura'''
    #Separating LA County properties and non-LA County properties
    la = train[train.county == 'los_angeles']
    not_la = train[train.county != 'los_angeles']
    result = kruskal(la.home_value, not_la.home_value)
    return result

def pearsonr_bedrooms(train):
    '''In this code, we use the pearsonr() function to calculate the Pearson correlation coefficient 
    between the bedrooms and home_value columns in the train DataFrame. The function returns two values: 
    the correlation coefficient and the p-value of the test. The output will show the correlation 
    coefficient and p-value.'''
    
    correlation, p_value = pearsonr(train['bedrooms'], train['home_value'])
    print("Correlation:", correlation)
    print("P-value:", p_value)
    
def pearsonr_bathrooms(train):
    '''In this function we use the pearsonr() test to calculate the Pearson correlation coefficient 
    between the bathrooms and home_value columns in the train DataFrame. The function returns two values: 
    the correlation coefficient and the p-value of the test. The output will show the correlation 
    coefficient and p-value.'''
    
    correlation, p_value = pearsonr(train['bathrooms'], train['home_value'])
    print("Correlation:", correlation)
    print("P-value:", p_value)

def pearsonr_sq_ft(train):
    '''In this funtion we use the pearsonr() test to calculate the Pearson correlation coefficient 
    between the square footage and home_value columns in the train DataFrame. The function returns two values: 
    the correlation coefficient and the p-value of the test. The output will show the correlation 
    coefficient and p-value.'''
    
    corr, p_value = pearsonr(train['sqft'], train['home_value'])

    print("Pearson correlation coefficient:", corr)
    print("P-value:", p_value)

#-------------------------------Vis------------------------------

def target_vis(train):
    '''This function takes in train and produces a displot from Seaborn
    utilizing the target variable, home_value'''
    #Set theme
    sns.set_theme()
    #Make the Plot
    target = sns.displot(train['home_value'],color='teal', height=6.5, aspect=15/8)
    target.set_axis_labels("Home Value", "Count of Properties")
    target.set(title='Distribution of Target Variable (Home Value)')
    # add a vertical line for the mean of the 'home_value' column
    mean_value = train['home_value'].mean()
    plt.axvline(mean_value, color='r', linestyle='--')
    plt.text(mean_value + 0.4, 1.2, 'Mean Home Value', rotation=90, va='center', color='r', fontsize=14, fontweight='bold')
    plt.show()


def county_vis(train):
    '''This function takes in train data and outputs a histplot visulaizing the difference in home value
    by county, with lines indicating mean tax value for properties'''
    # Binning Home Value
    train['brackets'] = pd.cut(train.home_value, 10, labels=[1,2,3,4,5,6,7,8,9,10])
    #Set theme
    sns.set_theme()
    #Set Palette
    palette = {
    'ventura': 'cyan',
    'orange': 'orange',
    'los_angeles': 'magenta'}
    # Set Size
    plt.figure(figsize=(12,8))
    # plot it
    sns.histplot(data=train, x='home_value', alpha=.8, hue='county', hue_order=['ventura', 'orange', 'los_angeles'], palette=palette)
    # add lines marking the mean value at each location
    plt.axvline(x=train[train.county == 'los_angeles'].home_value.mean(), color='magenta', linestyle='--')
    plt.axvline(x=train[train.county == 'orange'].home_value.mean(), color='orange', linestyle='--')
    plt.axvline(x=train[train.county == 'ventura'].home_value.mean(), color='cyan', linestyle='--')
    
    # axis tick labeling using
    plt.xticks(ticks = [0,200000,400000,600000,800000,1000000, 1200000], labels=['0', '200k', '400k', '600k', '800k', '1MM', '1.2MM'])
    
    # Make a title, label the axes
    plt.title('Home Value by County, With Dashed Lines Indicating County Mean Values')
    plt.xlabel('Home Value')
    plt.ylabel('Count of Homes')
    plt.show()

    
def get_hist_bedrooms(train):
    '''This function takes in train and returns a histogram visualizing the distribution
    of bedrooms'''
    fig, ax = plt.subplots()
    ax.hist(train.bedrooms, bins=[0, 1, 2, 3, 4, 5, 6], 
            color='lightsteelblue', label='bedrooms')
    plt.xlabel("Number of Bedrooms")
    plt.ylabel("Count of Homes")
    plt.title('Distribution of Bedrooms')
    plt.show()
    
    
def bedroom_vis(train):
    '''This function takes in training data and plots boxplots for bedrooms and bathrooms
    in comparison to Home Value using Seaborn Boxplots and Styling'''
    
    bedroom_groups = train.groupby('bedrooms')['home_value'].mean()
    
    sns.barplot(x=bedroom_groups.index, y=bedroom_groups.values, palette='flare')
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Average Home Value')
    plt.title('Average Home Value by Number of Bedrooms')
    plt.show()
    
def get_hist_bathrooms(train):
    '''This function teakes in train data & will output a histogram visualizing the 
    distribution of bedroom count'''
    
    fig, ax = plt.subplots()
    ax.hist(train.bathrooms, bins=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6], 
            color='lightsteelblue', label='bathrooms')
    plt.xlabel("Number of Bathrooms")
    plt.ylabel("Count of Homes")
    plt.title('Distribution of Bathrooms')
    plt.show()
    
    
def bathroom_vis(train):
    '''This function teakes in train data & will output a bar plot visualizing the 
    mean home value for each unique bedroom count'''
    
    bedroom_groups = train.groupby('bathrooms')['home_value'].mean()
    
    sns.barplot(x=bedroom_groups.index, y=bedroom_groups.values, palette='flare')
    plt.xlabel('Number of Bathrooms')
    plt.ylabel('Average Home Value')
    plt.title('Average Home Value by Number of Bathrooms')
    plt.show()
    
    
def get_hist_sqft(train):
    '''This function teakes in train data & will output a histogram visualizing the 
    distribution of square footage'''
    
    plt.hist(train['sqft'], bins=20, color='lightsteelblue')
    plt.xlabel('Square Footage')
    plt.ylabel('Count of Homes')
    plt.title('Distribution of Square Footage')
    plt.show()
    
    
def sqft_vis(train):
    '''This function teakes in train data & will output a barplot visualizing the 
    relationship between home value and square footage'''
    
    # create a new column with the binned square footage values
    train['sqft_bins'] = pd.cut(train.sqft, bins=30)

    # group the data by the binned square footage values and calculate the mean home value for each bin
    sqft_groups = train.groupby('sqft_bins')['home_value'].mean()

    # create a bar plot of the mean home value for each binned square footage range
    fig, ax = plt.subplots()
    sns.barplot(x=sqft_groups.index.astype(str), y=sqft_groups.values, color='lightsteelblue', ax=ax)
    ax.set_xlabel('Square Footage (Bins)')
    ax.set_ylabel('Mean Home Value')
    ax.set_title('Mean Home Value by Square Footage Bins')
    ax.set_xticklabels([])
    plt.show()

