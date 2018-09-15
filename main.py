#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 13:57:52 2018

@author: Dimasik007
"""
# path to the data files
# /Users/Dimasik007/Desktop/MSc_project/data/usdt_eth_2h.csv
# 1420070400 unix timestamp for 1st Jan 2015
# 1451606400 unix timestamp for 1st Jan 2016


import pandas as pd  # data analysis and manipulation
import numpy as np
# import matplotlib.pyplot as plt  # visualisation library
import seaborn as sns  # library for visualisation of statistical data
# from datetime import datetime
# from matplotlib import style
from collections import Counter
from sklearn import svm, cross_validation, neighbors  # cross_validation depreciated in favor of model_selection
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
# %matplotlib inline  # execute this in IPython console to put figuers inline
pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', 8)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 78)
pd.set_option('precision', 6)
sns.set()
# style.use('ggplot')

path = '/Users/Dimasik007/Desktop/Programming/PyCharm/Dissertation/svm_ga_system/data/'

# to read csv file and save it to a variable; set date as index
us_eth2h = pd.read_csv('/Users/Dimasik007/Desktop/MSc_project/data/usdt_eth_2h.csv',
                       index_col=['date'], parse_dates=True,
                       infer_datetime_format=True, dayfirst=True)


tickersUSD = ['USDT_BTC', 'USDT_BCH', 'USDT_ETC', 'USDT_XMR',
           'USDT_ETH', 'USDT_DASH', 'USDT_XRP', 'USDT_LTC',
           'USDT_NXT', 'USDT_STR', 'USDT_REP', 'USDT_ZEC']


# check pythonprogramming.net to make it more automated
def get_crypto_data(symbol, frequency, start=0, tocsv=False):
    '''
    Params: String symbol, int frequency in s = 300,900,1800,7200,14400,86400
    Returns: time series of ticker in a dataframe
    If tocsv=True creates a csv file, else returns df to assign to variable
    change needed fileName.csv to account for frequency in conditional
    '''
    url = 'https://poloniex.com/public?command=returnChartData&currencyPair=' \
          + symbol + '&end=9999999999&period=' + str(frequency) + '&start=' + str(start)
    df = pd.read_json(url)
    df.set_index('date', inplace=True)
    if tocsv:
        df.to_csv(symbol + '_daily.csv')  # add granularity before '.csv'
    else:
        return df
    print('Processed: ' + symbol)


# to download all tickrs and save to scv
for ticker in tickersUSD:
    get_crypto_data(ticker, frequency=86400, start=0, tocsv=True)


# combine all csv into one df with closing prices, fill NaN with 0
crypto_df = pd.DataFrame()
for ticker in tickersUSD:
    crypto_df[ticker] = pd.read_csv(path+ticker+'_daily.csv', index_col='date', parse_dates=True)['close']

crypto_df.fillna(0, inplace=True)  # fill NaN with 0


# assign ticker to a variable df
usdt_eth2h = get_crypto_data('USDT_ETH', 7200, 1451606400, tocsv=False)

'''
# building correlation table
crypto_corr = crypto_df['2017-06':'2018-06'].corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(14, 10))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(crypto_corr, xticklabels=crypto_corr.columns.values,
            yticklabels=crypto_corr.columns.values,
            square=True, linewidths=.5)

# visualise correlations between columns
sns.heatmap(crypto_corr,
            xticklabels=crypto_corr.columns.values,
            yticklabels=crypto_corr.columns.values)
'''


def process_data_for_labels(ticker):
    hm_days = 7
    # df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    df = crypto_df
    tickers = tickersUSD
    df.fillna(0, inplace=True)

    for i in range(1, hm_days+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df


def buy_sell_hold(*args):  # *args allows to pass any number of parameters and iterable
    cols = [c for c in args]
    requirement = 0.02  # required percent change
    for col in cols:
        if col > requirement:
            return 1  # buy
        if col < -requirement:
            return -1  # sell
    return 0  # hold / do nothing


def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map(buy_sell_hold, df['{}_1d'.format(ticker)],
                                                             df['{}_2d'.format(ticker)],
                                                             df['{}_3d'.format(ticker)],
                                                             df['{}_4d'.format(ticker)],
                                                             df['{}_5d'.format(ticker)],
                                                             df['{}_6d'.format(ticker)],
                                                             df['{}_7d'.format(ticker)]
                                                ))

    vals = df['{}_target'.format('USDT_BTC')].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    # replace infinite data with 0 (eg after pct_change with 0)
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X, y, df


def do_ml(ticker):

    X, y, df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,
                                                                         y,
                                                                         test_size=0.25)
    # defining classifiers used in voting classifier
    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])
    # fit input (X_train: pct_change) to target (Y_train: 1, 0, -1)
    clf.fit(X_train, y_train)  # train classifier
    confidence = clf.score(X_test, y_test)  # get accuracy score
    print('accuracy:', confidence)
    predictions = clf.predict(X_test)
    print('predicted class counts:', Counter(predictions))
    print()
    print()
    return confidence


do_ml('USDT_BCH')

# to save trained model if it works well
# with open('svc_model_prices.pickle', 'wb') as f:
#     pickle.dump(clf, f)
# pickle_in = open('svc_model_prices.pickle', 'rb')
# clf_from_pickle = pickle.load(pickle_in)


# calculate MAs on weighted average column and add them as columns to dataframe
us_eth2h['MA7'] = us_eth2h['weightedAverage'].rolling(window=7, min_periods=1).mean()
us_eth2h['MA30'] = us_eth2h['weightedAverage'].rolling(window=30, min_periods=1).mean()
us_eth2h['MA90'] = us_eth2h['weightedAverage'].rolling(window=90, min_periods=1).mean()
us_eth2h['MA120'] = us_eth2h['weightedAverage'].rolling(window=120, min_periods=1).mean()

# to plot MAs and weighted average price
us_eth2h['2017-01':'2018-06'][['weightedAverage', 'MA7', 'MA30', 'MA120']].plot(figsize=(14, 10))


"""
# plotting everything together___________________________________________________________________________________
# Prepare plot (4 subplots on 1 column)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
# Remove horizontal space between axes
fig.subplots_adjust(hspace=0)
fig.xlabel('Year', fontsize=25)

# set plot size
fig.set_size_inches(15, 30)

# plotting closing prices
ax1.set_ylabel('EUR / USD', size=25)
EUR_USD.Close.resample('M').mean().plot(ax=ax1, c='black')
# plotting moving averages
EUR_USD.sma.resample('M').mean().plot(ax=ax1, c='r', label='SMA30')
EUR_USD.ema.resample('M').mean().plot(ax=ax1, c='g', label='EMA30')
EUR_USD.wma.resample('M').mean().plot(ax=ax1, c='b', label='WMA30')
# setting legend
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels)

# plotting RSI subplot
ax2.set_ylabel('RSI', size=20)
EUR_USD.rsi.resample('M').mean().plot(ax=ax2, c='g', label='RSI 14')
ax2.axhline(y=30, c='b')
ax2.axhline(y=50, c='black')
ax2.axhline(y=70, c='b')
ax2.set_ylim([0, 100])
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels)

# plotting MACD
ax3.set_ylabel('MACD: 12, 26, 9', size=20)
EUR_USD.macd.resample('M').mean().plot(ax=ax3, color='b', label='Macd')
EUR_USD.macdSignal.resample('M').mean().plot(ax=ax3, color='g', label='Signal')
EUR_USD.macdHist.resample('M').mean().plot(ax=ax3, color='r', label='Hist')
ax3.axhline(0, lw=2, color='0')
handles, labels = ax3.get_legend_handles_labels()
ax3.legend(handles, labels)

# plotting ROC
ax4.set_ylabel('ROC: 10', size=20)
EUR_USD.roc.resample('M').mean().plot(ax=ax4, color='r', label='ROC')
handles, labels = ax4.get_legend_handles_labels()
ax4.legend(handles, labels)

# plt.show()_____________________________________________________________________________________________________



# TODO add labels on the graphs and find a way of plotting macd hist
f, (ax1, ax2, ax3, ax4) = plt.subplots( 4, 1, sharex=True, figsize=(30, 20))
f.subplots_adjust( hspace=0 )  # remove space between subplots

x1 = backtesting_data[['Close', 'SMA', 'EMA', 'WMA']]
ax1.plot(x1)
ax1.set_ylabel('EUR / USD Price', fontsize=23)

x2 = backtesting_data[['RSI']]
ax2.plot(x2)
ax2.set_ylabel('RSI', fontsize=23)
ax2.axhline(y=30, color='green')
ax2.axhline(y=70, color='red')

x3 = backtesting_data[['MACD', 'macdSignal']]
x4 = backtesting_data[['macdHist']]
ax3.plot(x3)
ax3.hist(x4)
ax3.set_ylabel('MACD 12, 26, 9', fontsize=23)

x5 = backtesting_data[['ROC']]
ax4.plot(x5)
ax4.axhline(y=0, color='grey')
ax4.set_ylabel('ROC', fontsize=23)
ax4.set_xlabel('Time', fontsize=25)

plt.show()
"""
