# stock_price_prediction 

## Overview
* linear regression model which predicts stock price based on 10 day, 50 day and 100 day moving averages
* optmized AdaBoost, Bagging and Random Forest regressors with GridCV


![alt text](https://github.com/sesankm/stock_prediction/blob/master/google_price_chart.png)

## Libraries used:
<strong>Python Version:</strong>3.8
<strong>Libraries:</strong>: pandas, scikit-learn, yahoo_fin, seaborn, matplotlib, numpy


## correlation heat map of features for google
* features used in regression models: 
	* 10 SMA
	* 50 SMA
	* Price v 100 SMA

![alt_text](https://github.com/sesankm/stock_price_prediction/blob/master/googl_correlation_heatmap.png)

# Models
* MAE of models trained with Google stock data
* <strong>Linear Regression</strong>: Negative MAE=-43.88
* <strong>AdaBoost Regressor</strong>: MAE=33.16
* <strong>Bagging Regressor</strong>: MAE=15.85
* <strong>Random Forest Regressor</strong> MAE=16.27

