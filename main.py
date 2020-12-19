from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from yahoo_fin.stock_info import *

def calcSMA(curr_index, window, df):
    shifted = np.full(window, 0).tolist() + df["close"].tolist()[:-window]
    return sum(shifted[curr_index-window: curr_index]) / window

def update_df_sma(window, df):
    sma =  np.full(window*2, np.nan).tolist() + [calcSMA(df.index.tolist().index(ind), window, df) for ind in df.index.tolist()[window*2:]]
    df["{} SMA".format(window)] = sma
    df["Price v {} SMA".format(window)] = (df["close"] - df["{} SMA".format(window)]) / df["{} SMA".format(window)]

def read_and_preproces(ticker):
    # read and clean company historical price data
    df = get_data(ticker, start_date="2018-01-01")
    df["prev_day_close"] = [np.nan] + df["close"].tolist()[:-1]
    df["prev_day_open"] = [np.nan] + df["open"].tolist()[:-1]
    df["prev_day_high"] = [np.nan] + df["high"].tolist()[:-1]
    update_df_sma(10, df)
    update_df_sma(50, df)
    update_df_sma(100, df)
    df.dropna(inplace=True)
    return df

def show_correlation_heatmap(df):
    # display heatmap
    sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, cmap="Blues")

def plot_price(ticker, df):
    # plot figure
    fig = plt.figure(figsize=(15, 10))
    spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[5, 1])
    ax0 = fig.add_subplot(spec[0])
    ax0.plot(df["close"])
    plt.xlabel("Date")
    plt.plot(df["10 SMA"])
    plt.plot(df["50 SMA"])
    plt.plot(df["100 SMA"])
    plt.legend([ticker, "10 SMA", "50 SMA", "100 SMA"])
    plt.ylabel("Price")
    plt.title(ticker)
    ax1 = fig.add_subplot(spec[1])
    ax1.bar(df.index, df["volume"])
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.show()

if __name__ == "__main__":
    ticker = input("Enter ticker: ")
    df = read_and_preproces(ticker)
    
    show_correlation_heatmap(df)
    plot_price(ticker, df)
    
    X = df[["10 SMA", "50 SMA", "Price v 100 SMA"]].to_numpy()
    y = df["close"].to_numpy()
    X *= .001

    # train test split    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.4, train_size=.6)
    
    # model building
    linear_regression_model = LinearRegression().fit(x_train, y_train)
    print(np.mean(cross_val_score(linear_regression_model, x_train, y_train, scoring = 'neg_mean_absolute_error')))
    
    # grid search
    # adaboost regressor
    adaboost_params = {"n_estimators": range(50, 300, 10), 
                       'loss':('linear', 'square', 'exponential')}
    adaboost_regressor_model = GridSearchCV(AdaBoostRegressor(), adaboost_params, scoring="r2").fit(x_train, y_train).best_estimator_
    adaboost_pred = adaboost_regressor_model.predict(x_test)
    print(mean_absolute_error(y_test, adaboost_pred))
    
    # bagging regressor
    bagging_params = {"n_estimators": range(10, 200, 10)}
    bagging_regressor_model = GridSearchCV(BaggingRegressor(), bagging_params, scoring='r2').fit(x_train, y_train).best_estimator_
    bagging_pred = bagging_regressor_model.predict(x_test)
    print(mean_absolute_error(y_test, bagging_pred))
    
    # random forrest regressor
    rf_parameters = {'n_estimators': range(100, 200, 10), 
                     'criterion': ('mse', 'mae'), 
                     'max_features': ('auto', 'sqrt', 'log2')}
    rf_regressor_model = GridSearchCV(RandomForestRegressor(), rf_parameters, scoring='r2').fit(x_train, y_train).best_estimator_
    rf_pred = rf_regressor_model.predict(x_test)
    print(mean_absolute_error(y_test, rf_pred))
    
    
    