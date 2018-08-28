import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
import math
import pandas as pd
import numpy as np
import talib as ta  # financial technical analysis lib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils import resample, class_weight
from sklearn.svm import SVC
from sklearn.externals import joblib  # to save model to pickle file
from sklearn.metrics import mean_squared_error

sns.set()  # set beautiful style for graphs

pd.options.mode.chained_assignment = None  # default='warn' disable warnings from pandas


def get_tis( df, dropnan=True, drop_vol=True ):
    """ calculate technical indicators with ta-lib
    requires a df with a column named "Close" """
    # to delete rows which have 0 volume / delete weekends
    print( "dropping rows with zero volume from dataFrame {}".format( df.shape ) )
    if drop_vol:
        df = df[ (df[ [ 'Volume' ] ] != 0).all( axis=1 ) ]
    print( "creating technical indicators" )
    df[ 'RSI' ] = ta.RSI( df.Close.values, timeperiod=14 )
    df[ 'ROC' ] = ta.ROC( df.Close.values, timeperiod=10 )
    df[ 'SMA' ] = ta.SMA( df.Close.values, timeperiod=30 )
    df[ 'EMA' ] = ta.EMA( df.Close.values, timeperiod=30 )
    df[ 'WMA' ] = ta.WMA( df.Close.values, timeperiod=30 )
    df[ 'MACD' ], df[ 'macdSignal' ], df[ 'macdHist' ] = ta.MACD( df.Close.values, fastperiod=12,
                                                                  slowperiod=26, signalperiod=9 )
    print( "done {}".format( df.shape ) )
    print( "creating market types" )
    # to label different market types ( 1 - Uptrend, 0 - Sideways, -1 - Downtrend )
    df[ 'mType' ] = np.where( df[ 'ROC' ] > 0.5, 1, (np.where( df[ 'ROC' ] < -0.5, -1, 0 )) )
    print( "done {}".format( df.shape ) )
    print( "dropping NaN values" )
    if dropnan:
        df.dropna( inplace=True )
    print( "returning dataFrame: {}  from get_tis func".format( df.shape ) )
    return df


def make_trading_rules( df ):
    """ generate trading rules based on technical indicators: [RSI, ROC, SMA, EMA, WMA and MACD] """
    print( "generating trading rules" )
    df = get_tis( df, dropnan=True, drop_vol=True )
    df.dropna( inplace=True )  # drop NaN values, in case there are some

    df[ 'R_RSI' ] = np.where( df[ 'RSI' ] > 70, -1, (np.where( df[ 'RSI' ] < 30, 1, 0 )) )
    df[ 'R_ROC' ] = np.where( df[ 'ROC' ] > 0, 1, (np.where( df[ 'ROC' ] < 0, -1, 0 )) )
    df[ 'R_SMA' ] = np.where( df[ 'SMA' ] < df[ 'Close' ], 1, (np.where( df[ 'SMA' ] > df[ 'Close' ], -1, 0 )) )
    df[ 'R_EMA' ] = np.where( df[ 'EMA' ] < df[ 'Close' ], 1, (np.where( df[ 'EMA' ] > df[ 'Close' ], -1, 0 )) )
    df[ 'R_WMA' ] = np.where( df[ 'WMA' ] < df[ 'Close' ], 1, (np.where( df[ 'WMA' ] > df[ 'Close' ], -1, 0 )) )
    df[ 'R_MACD' ] = np.where( df[ 'MACD' ] > 0, 1, (np.where( df[ 'MACD' ] < 0, -1, 0 )) )

    col_list = [ 'R_RSI', 'R_ROC', 'R_SMA', 'R_EMA', 'R_WMA', 'R_MACD' ]
    df[ 'Vote' ] = df[ col_list ].sum( axis=1 ) / len( col_list )
    # TODO try different trading rules, eg buy/sell when ROC +/-0.15 etc.

    return df


def normalise( x ):
    """ standardize features by removing the mean and scaling to unit variance """
    scaler = StandardScaler()
    x_norm = scaler.fit_transform( x.values )
    x_norm = pd.DataFrame( x_norm, index=x.index, columns=x.columns )
    print( "normalised data {}".format( x_norm.shape ) )
    return x_norm


def create_features_labels( df, price_seq=True, hm_lags=99 ):
    """
    prepare dataset fot machine learning
    if price_seq==True, return df with hm_lags price sequences and market type shifted by 1
    else returns a dataFrame with sequences of technical indicators and market type shifted by 1
    """
    print( "create features and labels" )
    df = get_tis( df, dropnan=True, drop_vol=True )
    if price_seq:  # type 1 - use hm_lags price sequences to predict market next day
        print( "creating dataset with price sequences" )
        for i in range( 0, hm_lags ):
            df[ "Close_{}".format( str( i + 1 ) ) ] = df[ "Close" ].shift( i + 1 )
        print( "done, df: {}; dropping extra columns next".format( df.shape ) )
        df[ 'mType1' ] = df[ 'mType' ].shift( 1 ).fillna( 9 ).astype( int )  # create shifted market type column
        df = df[ (df[ [ 'mType1' ] ] != 9).all( axis=1 ) ]  # drop rows left after shifting market type
        df.dropna( inplace=True )  # drop NaN values left after shifting values
        df.drop( [ 'RSI', 'ROC', 'SMA', 'EMA', 'WMA', 'MACD', 'macdSignal', 'macdHist', 'Volume', 'mType' ], axis=1,
                 inplace=True )  # drop TI and previous market type
        print( "returning price sequences: {} from create features and labels func".format( df.shape ) )
        return df
    else:  # type 2 - use technical indicators to predict market next day
        print( "creating dataset with Technical Indicators" )
        df[ 'mType1' ] = df[ 'mType' ].shift( 1 ).fillna( 9 ).astype( int )  # create shifted market type column
        df = df[( df[[ 'mType1' ]] != 9).all(axis=1)]  # drop rows left after shifting market type
        df.drop( [ 'Close', 'Volume', 'mType' ], axis=1, inplace=True )
        df.dropna( inplace=True )
        print( "returning TIs and market type from create features and labels func {}".format( df.shape ) )
        return df


# machine learning here________________________________________________________________________________________
def get_data_for_ml( df, use_prices=True, start_testing=None, end_testing='2006-01-01', validation_start='2006-01-02',
                     validation_end=None ):
    """
    creates training, testing and validation sets to input in machine learning
    :param df: dataFrame with closing prices column
    :param use_prices: True to use price sequences and false to use technical indicators
    :param start_testing: beginning of testing period
    :param end_testing: end of testing period
    :param validation_start: validation period for backtesting
    :param validation_end: end of validation period
    :return: training, testing and validation sets
    """
    if use_prices:  # predict future market direction with price sequences
        print( "preparing price sequences data for machine learning" )
        data_for_ml = create_features_labels( df, price_seq=True, hm_lags=99 )
        data_train = data_for_ml[ start_testing:end_testing ]  # create training dataFrame
        y = data_train[ 'mType1' ].values  # target which we predict ( .values to transform it to ndarray )
        X = data_train.drop( [ 'mType1' ], axis=1 )  # drop target column
        X = normalise( X )  # normalise X ( leads to x3 accuracy improvement )
        X = X.values  # change type to ndarray
    else:  # predict future market direction with technical indicators
        print( "preparing TIs data for machine learning" )
        data_for_ml = create_features_labels( df, price_seq=False )
        data_train = data_for_ml[ start_testing:end_testing ]
        y = data_train[ 'mType1' ].values
        X = data_train.drop( [ 'mType1' ], axis=1 )  # data TI type 2
        X = normalise( X )
        X = X.values

    # create validation set
    print( "creating validation set" )
    data_val = data_for_ml[ validation_start:validation_end ]
    if use_prices:
        y_val = data_val[ 'mType1' ].values
        X_val = data_val.drop( [ 'mType1' ], axis=1 )
        X_val = normalise( X_val )
        X_val = X_val.values
    else:
        y_val = data_val[ 'mType1' ].values
        X_val = data_val.drop( [ 'mType1' ], axis=1 )  # use TI type 2
        X_val = normalise( X_val )
        X_val = X_val.values

    # splitting data for training and testing
    print( "splitting data for training and testing sets" )
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, shuffle=False )
    print( "Training data: {} and target: {}".format( X_train.shape, y_train.shape ) )
    print( "Testing data: {} and target: {}".format( X_test.shape, y_test.shape ) )

    # adjust weights of the classes to account for unbalanced data
    class_weights = class_weight.compute_class_weight( 'balanced', np.unique( y_train ), y_train )

    # check the sum of each value
    unique, counts = np.unique( y, return_counts=True )
    print( "Counts of labels" )
    print( dict( zip( unique, counts ) ) )
    # use that in class_weight parameter of the classifier, as it accepts only dict
    print( "Class weights: {}".format( class_weights ) )

    return X_train, X_test, y_train, y_test, X_val, y_val


# load csv file to df
EU = pd.read_csv( 'data/EURUSD_1H.csv', index_col=[ 'Time' ], usecols=[ 'Time', 'Close', 'Volume' ], parse_dates=True,
                  infer_datetime_format=True, dayfirst=True )
EUR_USD = EU.copy()

# prepare type 1 data
X_train, X_test, y_train, y_test, X_val, y_val = get_data_for_ml( df=EUR_USD, use_prices=True,
                                                                  start_testing='2011-01-01', end_testing='2016-01-01',
                                                                  validation_start='2016-01-02', validation_end=None )

# prepare type 2 data
X_train, X_test, y_train, y_test, X_val, y_val = get_data_for_ml( df=EUR_USD, use_prices=False,
                                                                  start_testing='2007-01-01', end_testing='2016-01-01',
                                                                  validation_start='2016-01-02', validation_end=None )

# specify parameters to try for classifier
svc_parameters = [ {'kernel': ['poly'], 'degree': [2, 3, 4], 'C': [1, 10, 100, 1000], 'gamma': [ 0.001, 0.01, 0.1 ] },
                   {'kernel': ['rbf'], 'C': [1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1]},
                   {'kernel': ['linear'], 'C': [1, 10, 100, 1000]} ]  # DON'T USE gamma=1 in poly kernel and 0.0001

# do exhaustive search for best parameters to find those that max accuracy score and do K-fold with K=5
# manually input class weights based on the output from get_data_for_ml function
clf = GridSearchCV( SVC( class_weight={ -1: 4.81567265, 0: 0.38437245, 1: 5.24379992 }, cache_size=400 ),
                    param_grid=svc_parameters, cv=5, scoring='accuracy', n_jobs=-1, refit=True,
                    return_train_score=False, verbose=42 )

# fit input (X_train: price sequences / TIs) to target (y_train: 1, 0, -1)
clf.fit( X_train, y_train )

# print results
print( "Accuracy on training data {0:0.2f} %".format( clf.score( X_train, y_train ) ) )
print( "Best parameters set found on development set:" )
print( clf.best_params_ )
print()
print( "Grid scores on training set:" )
means = clf.cv_results_[ 'mean_test_score' ]
stds = clf.cv_results_[ 'std_test_score' ]
for mean, std, params in zip( means, stds, clf.cv_results_[ 'params' ] ):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

# predict testing dataset
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('Support Vector Machine Classifier\n {}\n'.format(classification_report(y_test, y_pred,
                                                                              target_names=['Downtrend',
                                                                                            'Sideways',
                                                                                            'Uptrend'])))
print("Accuracy Score: {0:0.2f} %".format(acc * 100))
MSE = mean_squared_error( y_test, y_pred )
print( "Mean Squared Error (MSE): {}".format( MSE ) )
print( "Root Mean Square Error (RMSE): {}".format( math.sqrt( MSE ) ) )
print( "Best score achieved by classifier: {}".format( clf.best_score_ ) )
print(clf.best_estimator_)
# compress is used to put all the files into a single pickle file
joblib.dump(clf.best_estimator_, 'svc_ti.pkl', compress=1)  # save model trained on TI to pkl file
joblib.dump( clf.best_estimator_, 'svc_prices_mType.pkl',
             compress=1 )  # save model trained on price sequences to pkl file

clf2 = joblib.load( 'svc_prices_mType.pkl' )  # load classifier to a variable


# predict validation set
y_pred2 = clf2.predict(X_val)
acc2 = accuracy_score(y_val, y_pred2)
print('Support Vector Machine Classifier\n {}\n'.format(classification_report(y_val, y_pred2,
                                                                              target_names=['Downtrend',
                                                                                            'Sideways',
                                                                                            'Uptrend'])))
print("Accuracy Score: {0:0.2f} %".format(acc2 * 100))
MSE_val = mean_squared_error( y_val, y_pred2 )
print( "Mean Squared Error (MSE): {}".format( MSE_val ) )
print( "Root Mean Square Error (RMSE): {}".format( math.sqrt( MSE_val ) ) )

# do trading__________________________________________________________________________________________
df_for_trading = make_trading_rules( EUR_USD )  # create dataset for trading

backtesting_data = df_for_trading[ '2016-01-02': ]  # leave only needed time frame
backtesting_data[ 'm_pred' ] = y_pred2  # attach market direction predictions from ml


def do_backtest( data ):
    """
    function allows to iterate through a dataFrame and place trades according to trading rules.
    at the end it outputs updated dataFrame and order book with all the trades performed
    substitute commission for different assets
    """

    equity_amount = 100000  # set the amount of capital to trade with
    open_order = 0
    fx_commission = 1  # 1$ per trade
    order_book = pd.DataFrame()
    data[ 'Equity' ] = 0
    data[ 'Currency' ] = 'currency'
    # crypto_commission = 0.001  # 0.10 % e.g. on Poloniex ( - equity_amount * crypto_commission )

    data[ 'Trade' ] = np.where( (data[ 'Vote' ] >= 0.8) & (data[ 'm_pred' ] == 1), 1,
                                np.where( (data[ 'Vote' ] <= -0.8) & (data[ 'm_pred' ] == -1), -1, 0 ) )

    for index, row in data.iterrows():

        if row[ 'Trade' ] == 1 and open_order == 0:  # check buy signal and if there's no open order
            open_order = 1  # open order and buy with formula below
            equity_amount = (equity_amount / data.loc[ index, 'Close' ]) - fx_commission
            data.at[ index, 'Equity' ] = equity_amount  # record amount of money we have
            data.at[ index, 'Currency' ] = 'EUR'
            order_book = data.loc[ :, [ 'Close', 'Vote', 'm_pred', 'Trade', 'Equity', 'Currency' ] ]

        elif row[ 'Trade' ] == -1 and open_order == 1:  # check sell signal
            open_order = 0  # close previously opened order and sell with below formula
            equity_amount = (equity_amount * data.loc[ index, 'Close' ]) - fx_commission
            data.at[ index, 'Equity' ] = equity_amount  # record amount of money we have
            data.at[ index, 'Currency' ] = 'USD'
            order_book = data.loc[ :, [ 'Close', 'Vote', 'm_pred', 'Trade', 'Equity', 'Currency' ] ]

    order_book = order_book[ (order_book[ [ 'Equity' ] ] != 0).all( axis=1 ) ]

    performance = order_book[ (order_book[ [ 'Trade' ] ] != 1).all( axis=1 ) ]

    # plotting performance and prices
    f, (ax1, ax2) = plt.subplots( 2, 1, sharex=True, figsize=(30, 15) )  # figure with 2 subplots and shared x axis
    f.subplots_adjust( hspace=0 )  # remove space between subplots
    x1 = data[ [ 'Close' ] ]  # .resample('W' ).mean()  # plot EUR/USD price resampled weekly
    ax1.plot( x1 )
    ax1.set_title( 'System Performance', fontsize=23 )
    ax1.set_ylabel( 'EUR / USD Price', fontsize=20 )

    x2 = performance[ [ 'Equity' ] ]
    ax2.set_ylabel( 'System Returns in $', fontsize=20 )
    ax2.set_xlabel( 'Years', fontsize=25 )
    ax2.plot( x2 )

    plt.show()

    return data, order_book, equity_amount


backtested_df, new_order_book, eq = do_backtest( backtesting_data )



"""
def rebalance(unbalanced_data):

    # Separate majority and minority classes
    data_minority = unbalanced_data[unbalanced_data.target == 0]
    data_majority = unbalanced_data[unbalanced_data.target == 1]

    # Upsample minority class
    n_samples = len(data_majority)
    data_minority_upsampled = resample(data_minority, replace=True, n_samples=n_samples, random_state=5)

    # Combine majority class with upsampled minority class
    data_upsampled = pd.concat([data_majority, data_minority_upsampled])

    data_upsampled.sort_index(inplace=True)

    # Display new class counts
    data_upsampled.target.value_counts()

    return data_upsampled
"""

"""
clf = OneVsRestClassifier(SVC(kernel='rbf', C=10, gamma=0.001))
clf.fit(X_train, y_train)
clf.score(X_train, y_train)
"""

"""
# do cross validation
Kcv = KFold(n_splits=5)
prediction_cv = cross_val_score(clf, X, y, cv=cv)
"""


# plotting TIs and closing price on one plot
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20, 10))  # create figure with 2 subplots with shared x axis
f.subplots_adjust(hspace=0)  # remove space between subplots
x1 = EUR_USD[ '2017-01-01':'2018-06-01' ][ [ 'Close' ] ].resample( 'W' ).mean()  # plot EUR/USD price resampled weekly
ax1.plot(x1)
ax1.set_title('EUR/USD price and ROC indicator', fontsize=16)
ax1.set_ylabel('EUR/USD price', fontsize=12)

x2 = EUR_USD[ '2017-01-01':'2018-06-01' ][ [ 'ROC' ] ].resample( 'W' ).mean()
ax2.set_ylabel( '9 ROC', fontsize=12 )
ax2.set_xlabel( 'Year', fontsize=16 )
ax2.plot(x2)

plt.show()

# plotting
"""
# to plot specific time period
# EUR_USD['2017-01-01':'2018-06-01'][['close']].plot()
# plt.show()

# Prepare plot (4 subplots on 1 column)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True)
# Remove horizontal space between axes
fig.subplots_adjust(hspace=0)

# set plot size
fig.set_size_inches(15, 30)

# plotting closing prices
ax1.set_ylabel('EUR / USD', size=20)
EUR_USD.Close.resample('M').mean().plot(ax=ax1, c='black')
# plotting moving averages
EUR_USD.SMA.resample('M').mean().plot(ax=ax1, c='r', label='SMA30')
EUR_USD.EMA.resample('M').mean().plot(ax=ax1, c='g', label='EMA30')
EUR_USD.WMA.resample('M').mean().plot(ax=ax1, c='b', label='WMA30')
# setting legend
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels)

# plotting RSI subplot
ax2.set_ylabel('RSI', size=20)
EUR_USD.RSI.resample('M').mean().plot(ax=ax2, c='g', label='RSI 14')
ax2.axhline(y=30, c='b')
ax2.axhline(y=50, c='black')
ax2.axhline(y=70, c='b')
ax2.set_ylim([0, 100])
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels)

# plotting MACD
ax3.set_ylabel('MACD: 12, 26, 9', size=20)
EUR_USD.MACD.resample('M').mean().plot(ax=ax3, color='blue', label='Macd')
EUR_USD.macdSignal.resample('M').mean().plot(ax=ax3, color='green', label='Signal')
EUR_USD.macdHist.resample('M').mean().hist(ax=ax3, bins=50, color='grey', label='Hist')
ax3.axhline(0, lw=2, color='0')
handles, labels = ax3.get_legend_handles_labels()
ax3.legend(handles, labels)

# plotting ROC
ax4.set_ylabel('ROC: 10', size=20)
ax4.set_xlabel('Year', size=25)
EUR_USD.ROC.resample('M').mean().plot(ax=ax4, color='r', label='ROC')
handles, labels = ax4.get_legend_handles_labels()
ax4.legend(handles, labels)

plt.show()    ___________________________________________________________________________________________________
"""

# plotting market types and closing price together
"""
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(20, 10))
fig.subplots_adjust(hspace=0)
ax1.set_ylabel('EUR / USD')
EUR_USD['2015-01-01':'2017-01-01'][['Close']].plot(ax=ax1, c='g')
ax2.set_ylabel('Market Type')
EUR_USD['2015-01-01':'2017-01-01'][['mType']].plot(ax=ax2)
plt.show()
"""


def make_lags( df, max_lag, min_lag=0, separator='_' ):
    """ shift values in given columns of dataFrame """
    data = [ ]
    for i in range( min_lag, max_lag + 1 ):
        data.append( df.shift( i ).copy() )
        data[ -1 ].columns = [ c + separator + str( i ) for c in df.columns ]
    print( "returning values from make_lags func {}" )
    return pd.concat( data, axis=1 )

# doesn't work now
# alternative implementation to speed up classification process by
# training several classifiers on different subsets of data
n_estimators = 10
clf3 = GridSearchCV( OneVsRestClassifier(
    BaggingClassifier( SVC( class_weight={ -1: 3.48339912, 0: 0.41015337, 1: 3.6388552 }, cache_size=400 ),
                       max_samples=1.0 / n_estimators, n_estimators=n_estimators, bootstrap=False ) ),
                     param_grid=svc_parameters, cv=5, scoring='accuracy', n_jobs=-1, refit=True,
                     return_train_score=False, verbose=42 )

grid = ParameterGrid( { "max_samples":        [ 0.5, 1.0 ], "max_features": [ 1, 2, 4 ], "bootstrap": [ True, False ],
                        "bootstrap_features": [ True, False ] } )

for base_estimator in [ None, DummyClassifier(), Perceptron(), DecisionTreeClassifier(), KNeighborsClassifier(),
                        SVC() ]:
    for params in grid:
        BaggingClassifier( base_estimator=base_estimator, random_state=rng, **params ).fit( X_train, y_train ).predict(
                X_test )

    ######################
    bsh_cond = [ (data[ 'Vote' ] >= 0.8) & (data[ 'm_pred' ] == 1),
                 (data[ 'Vote' ] <= -0.8) & (data[ 'm_pred' ] == -1) ]
    choices = [ 1, -1 ]
    data[ 'Trade' ] = np.select( bsh_cond, choices )

    # SUBSTITUTE COMMISSIONS ACCORDINGLY
    data[ 'Equity' ] = np.where( data[ 'Trade' ] == 1, (equity_amount / data[ 'Close' ] - fx_commission), (
        np.where( data[ 'Trade' ] == -1, (equity_amount * data[ 'Close' ] - fx_commission), equity_amount )) )
    data[ 'Eq_change' ] = data[ 'Equity' ].pct_change().fillna( 0 )

if (row.Trade == 1) & (row.pos == 0):
    row.pos = row.replace( { 'pos': 1 } )
    row.Equity = (equity_amount / row.Close) - fx_commission  # substitute commissions accordingly
    equity_amount = row.Equity
    order_book = data.loc[ :, [ 'Close', 'Trade', 'Equity' ] ]
elif (row.Trade == -1) & (row.pos == 1):
    row.pos = row.replace( { 'pos': 0 } )
    row.Equity = (equity_amount * row.Close) - fx_commission  # substitute commissions accordingly
    equity_amount = row.Equity
    order_book = data.loc[ :, [ 'Close', 'Trade', 'Equity' ] ]
