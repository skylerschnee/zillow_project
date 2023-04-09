# Zillow Regression Project  

# Project Description:
Within this project we will wrangle zillow data, identify significant features, and build a regression model to predict single-family home values.

# Project Goal:
* Identify drivers of single family homes.
* Use drivers to develop machine learning models to predict home value.

# Initial Thoughts:
My initial hypothesis is that county, bedrooms, bathrooms, & square feet, are significant drivers of home value.

# The Plan
* Aquire data from codeup database

* Prepare data

* Discover potential drivers of home value through exploration

  * Answer these questions:
    * Does location significanty influence home value?
    * Does bedroom count significanty influence value? 
    * Does bathroom count significanty influence value?
    * Does square footage significanty influence value?
  
* Develop a Model to predict tax value

    * Use drivers identified through exploration to build predictive models of different types
    * Evaluate models on train and validate data
    * Select the best model based on lowest RMSE in combination with R2 scores
    * Evaluate the best model on test data
    
* Draw conclusions



# Data Dictionary

| Feature | Definition |
| --- | --- |
| home_Value | The tax assesor appraised home value (Target)|
| bedrooms | Number of bedrooms for the home |
| bathrooms | Number of bathrooms for the home |
| sqft | Total square footage listed for a property |
| county | The county the home is located within |
| year_built | Year the home was built |

# Steps to Reproduce

1. Clone this repository
2. Acquire the data from the Codeup Database ('Zillow')
3. Put the data in the file containing the cloned repo.
4. Create or copy your env.py file to this repo, specifying the codeup hostname, username and password
5. Run notebook.

# Takeaways and Conclusions

- All models performed better than the baseline
- The RMSE for OLS was the lowest at 223504
- Because of this RMSE score and the low RMSE delta between train and validate set, I will proceed with this model on my test set.
- Our test data results were similar to train and validate
- The OLS model has improved the accuracy of the predictions by reducing the error from the baseline by approximately 17%.

# Recommendations

- If deeper location data were available like zip code for example, we could have much stronger of a location driver than just county

# Next Steps

- With more time, I would like to do more feature engineering, run addiitonal regression models, and explore the original database futher to see what features can be added