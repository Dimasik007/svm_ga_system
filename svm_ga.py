import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
import math
import pandas as pd
import numpy as np
import talib as ta  # financial technical analysis lib
import seaborn as sns
import matplotlib

matplotlib.rc( 'xtick', labelsize=22 )  # set size of the axis font
matplotlib.rc( 'ytick', labelsize=22 )
# from matplotlib import style || # style.use('ggplot')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.svm import SVC
from sklearn.externals import joblib  # to save model to pickle file
from sklearn.metrics import mean_squared_error

sns.set()  # set beautiful style for graphs
pd.options.mode.chained_assignment = None  # default='warn' disable warnings from pandas

def get_crypto_data( symbol, frequency, start=0, tocsv=False ):
    '''
    Params: String symbol, int frequency in s = 300,900,1800,7200,14400,86400
    Returns: time series of ticker in a dataFrame (close, high, low, open, quoteVolume, volume, weightedAverage
    If tocsv=True creates a csv file, else returns df to assign to variable
    change needed fileName.csv to account for frequency in conditional
    '''
    url = 'https://poloniex.com/public?command=returnChartData&currencyPair=' + symbol + '&end=9999999999&period=' + str(
        frequency ) + '&start=' + str( start )
    df = pd.read_json( url )
    df.set_index( 'date', inplace=True )
    if tocsv:
        df.to_csv( symbol + '_daily.csv' )  # add granularity before '.csv'
    else:
        return df
    print( 'Processed: ' + symbol )


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
    change bsh_rule column values depending on the assets and their volatility e.g. 0.001 to 0.002
    """
    print( "create features and labels" )
    df = get_tis( df, dropnan=True, drop_vol=True )

    # getting price percent changes in the next hour
    df[ 'Close_pctch' ] = df.Close.pct_change().shift( -1 )

    print( "creating market directions" )
    # to label different market types ( 1 - Uptrend, 0 - Sideways, -1 - Downtrend )
    df[ 'bsh_rule' ] = np.where( df[ 'Close_pctch' ] > 0.001, 1, (np.where( df[ 'Close_pctch' ] < -0.001, -1, 0 )) )
    print( "done {}".format( df.shape ) )

    if price_seq:  # type 1 - use hm_lags price sequences to predict market next day
        print( "creating dataset with price sequences" )
        for i in range( 0, hm_lags ):
            df[ "Close_{}".format( str( i + 1 ) ) ] = df[ "Close" ].shift( i + 1 )

        print( "done, df: {}; dropping extra columns next".format( df.shape ) )

        df.dropna( inplace=True )  # drop NaN values left after shifting values
        df.drop( [ 'RSI', 'ROC', 'SMA', 'EMA', 'WMA', 'MACD', 'macdSignal', 'macdHist', 'Volume' ], axis=1,
                 inplace=True )  # drop TI and previous market type

        print( "returning price sequences: {} from create features and labels func".format( df.shape ) )
        return df

    else:  # type 2 - use technical indicators to predict market next day
        print( "creating dataset with Technical Indicators" )
        df.drop( [ 'Close', 'Volume' ], axis=1, inplace=True )
        df.dropna( inplace=True )
        print( "returning TIs and price changes from create features and labels func {}".format( df.shape ) )
        return df


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
        y = data_train[ 'bsh_rule' ].values  # target which we predict ( .values to transform it to ndarray )
        X = data_train.drop( [ 'bsh_rule' ], axis=1 )  # drop target column
        X = normalise( X )  # normalise X ( leads to x3 accuracy improvement )
        X = X.values  # change type to ndarray

    else:  # predict future market direction with technical indicators
        print( "preparing TIs data for machine learning" )
        data_for_ml = create_features_labels( df, price_seq=False )
        data_train = data_for_ml[ start_testing:end_testing ]
        y = data_train[ 'bsh_rule' ].values
        X = data_train.drop( [ 'bsh_rule' ], axis=1 )  # data TI type 2
        X = normalise( X )
        X = X.values

    # create validation set
    print( "creating validation set" )
    data_val = data_for_ml[ validation_start:validation_end ]
    if use_prices:
        y_val = data_val[ 'bsh_rule' ].values
        X_val = data_val.drop( [ 'bsh_rule' ], axis=1 )
        X_val = normalise( X_val )
        X_val = X_val.values
    else:
        y_val = data_val[ 'bsh_rule' ].values
        X_val = data_val.drop( [ 'bsh_rule' ], axis=1 )  # use TI type 2
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


def do_backtest( data ):
    """
    function allows to iterate through a dataFrame and place trades according to trading rules.
    at the end it outputs updated dataFrame and order book with all the trades performed
    substitute commission for different assets
    """

    equity_amount = 100000  # set the amount of capital to trade with
    equity_amount_TI = 100000  # set the amount of capital to trade with
    equity_amount_svc = 100000  # set the amount of capital to trade with
    open_order = 0
    open_order_TI = 0
    open_order_svc = 0
    commission = 4
    # commission = 0.001  # 0.1 % commision on Poloniex and Dukascopy exchanges
    order_book = pd.DataFrame()
    order_book_TI = pd.DataFrame()
    order_book_svc = pd.DataFrame()
    data[ 'Equity' ] = 0.00  # to make values floats
    data[ 'Equity_TI' ] = 0.00  # to make values floats
    data[ 'Equity_svc' ] = 0.00  # to make values floats
    data[ 'Currency' ] = 'currency'

    # combining trading rules together
    data[ 'Trade' ] = np.where( (data[ 'Vote' ] >= 0.8) & (data[ 'm_pred' ] == 1), 1,
                                np.where( (data[ 'Vote' ] <= -0.8) & (data[ 'm_pred' ] == -1), -1, 0 ) )

    # trading rule using only Technical Indicators
    data[ 'Trade_TI' ] = np.where( data[ 'Vote' ] >= 0.8, 1, np.where( data[ 'Vote' ] <= -0.8, -1, 0 ) )

    # trading rule using only svc
    data[ 'Trade_svc' ] = np.where( data[ 'm_pred' ] == 1, 1, np.where( data[ 'm_pred' ] == -1, -1, 0 ) )

    for index, row in data.iterrows():  # using svc and Technical Indiccators

        if row[ 'Trade' ] == 1 and open_order == 0:  # check buy signal and if there's no open order
            open_order = 1  # open order and buy with formula below
            # commission_payable = equity_amount * commission  # commission for cryptocurrencies
            # equity_amount = ((equity_amount - commission_payable) / data.loc[ index, 'Close' ])
            equity_amount = (equity_amount - commission) / data.loc[ index, 'Close' ]  # with FX commission
            data.at[ index, 'Equity' ] = equity_amount  # record amount of money we have
            data.at[ index, 'Currency' ] = 'EUR'  # change to ETH, BTC, EUR, etc.
            order_book = data.loc[ :, [ 'Close', 'Vote', 'm_pred', 'Trade', 'Equity', 'Currency' ] ]

        elif row[ 'Trade' ] == -1 and open_order == 1:  # check sell signal
            open_order = 0  # close previously opened order and sell with below formula
            # commission_payable = equity_amount * commission
            # equity_amount = ((equity_amount - commission_payable) * data.loc[ index, 'Close' ])
            equity_amount = (equity_amount - commission) * data.loc[ index, 'Close' ]  # with FX commission
            data.at[ index, 'Equity' ] = equity_amount  # record amount of money we have
            data.at[ index, 'Currency' ] = 'USD'
            order_book = data.loc[ :, [ 'Close', 'Vote', 'm_pred', 'Trade', 'Equity', 'Currency' ] ]

    for index, row in data.iterrows():  # using only Technical Indicators

        if row[ 'Trade_TI' ] == 1 and open_order_TI == 0:  # check buy signal and if there's no open order
            open_order_TI = 1  # open order and buy with formula below
            # commission_payable = equity_amount * commission  # commission for cryptocurrencies
            # equity_amount = ((equity_amount - commission_payable) / data.loc[ index, 'Close' ])
            equity_amount_TI = (equity_amount_TI - commission) / data.loc[ index, 'Close' ]  # with FX commission
            data.at[ index, 'Equity_TI' ] = equity_amount_TI  # record amount of money we have
            data.at[ index, 'Currency' ] = 'EUR'  # change to ETH, BTC, EUR, etc.
            order_book_TI = data.loc[ :, [ 'Close', 'Vote', 'Trade_TI', 'Equity_TI', 'Currency' ] ]

        elif row[ 'Trade_TI' ] == -1 and open_order_TI == 1:  # check sell signal
            open_order_TI = 0  # close previously opened order and sell with below formula
            # commission_payable = equity_amount * commission
            # equity_amount = ((equity_amount - commission_payable) * data.loc[ index, 'Close' ])
            equity_amount_TI = (equity_amount_TI - commission) * data.loc[ index, 'Close' ]  # with FX commission
            data.at[ index, 'Equity_TI' ] = equity_amount_TI  # record amount of money we have
            data.at[ index, 'Currency' ] = 'USD'
            order_book_TI = data.loc[ :, [ 'Close', 'Vote', 'Trade_TI', 'Equity_TI', 'Currency' ] ]

    for index, row in data.iterrows():  # using only SVC

        if row[ 'Trade_svc' ] == 1 and open_order_svc == 0:  # check buy signal and if there's no open order
            open_order_svc = 1  # open order and buy with formula below
            # commission_payable = equity_amount * commission  # commission for cryptocurrencies
            # equity_amount = ((equity_amount - commission_payable) / data.loc[ index, 'Close' ])
            equity_amount_svc = (equity_amount_svc - commission) / data.loc[ index, 'Close' ]  # with FX commission
            data.at[ index, 'Equity_svc' ] = equity_amount_svc  # record amount of money we have
            data.at[ index, 'Currency' ] = 'EUR'  # change to ETH, BTC, EUR, etc.
            order_book_svc = data.loc[ :, [ 'Close', 'm_pred', 'Trade_svc', 'Equity_svc', 'Currency' ] ]

        elif row[ 'Trade_svc' ] == -1 and open_order_svc == 1:  # check sell signal
            open_order_svc = 0  # close previously opened order and sell with below formula
            # commission_payable = equity_amount * commission
            # equity_amount = ((equity_amount - commission_payable) * data.loc[ index, 'Close' ])
            equity_amount_svc = (equity_amount_svc - commission) * data.loc[ index, 'Close' ]  # with FX commission
            data.at[ index, 'Equity_svc' ] = equity_amount_svc  # record amount of money we have
            data.at[ index, 'Currency' ] = 'USD'
            order_book_svc = data.loc[ :, [ 'Close', 'm_pred', 'Trade_svc', 'Equity_svc', 'Currency' ] ]

    order_book = order_book[ (order_book[ [ 'Equity' ] ] != 0).all( axis=1 ) ]  # to record only trades
    order_book_TI = order_book_TI[ (order_book_TI[ [ 'Equity_TI' ] ] != 0).all( axis=1 ) ]  # to record only trades
    order_book_svc = order_book_svc[ (order_book_svc[ [ 'Equity_svc' ] ] != 0).all( axis=1 ) ]  # to record only trades

    performance = order_book[ (order_book[ [ 'Trade' ] ] != 1).all( axis=1 ) ]  # to track only values in USD
    performance_TI = order_book_TI[
        (order_book_TI[ [ 'Trade_TI' ] ] != 1).all( axis=1 ) ]  # to track only values in USD
    performance_svc = order_book_svc[
        (order_book_svc[ [ 'Trade_svc' ] ] != 1).all( axis=1 ) ]  # to track only values in USD

    # TODO add winning/losing trades, avg profit/loss per trade
    # sharpe = 10  # TODO add Sharp Ratio or Sortino ratio
    n_trades = len( order_book.index )  # get number of trades performed
    ann_return = performance.Equity.iloc[ -1 ]  # get last value of the equity column in USD
    pct_ann_return = (ann_return - 100000) / 100000 * 100

    # sharpe = 10  # TODO add Sharp Ratio or Sortino ratio
    n_trades_TI = len( order_book_TI.index )  # get number of trades performed
    ann_return_TI = performance_TI.Equity_TI.iloc[ -1 ]  # get last value of the equity column in USD
    pct_ann_return_TI = (ann_return_TI - 100000) / 100000 * 100

    # sharpe = 10  # TODO add Sharp Ratio or Sortino ratio
    n_trades_svc = len( order_book_svc.index )  # get number of trades performed
    ann_return_svc = performance_svc.Equity_svc.iloc[ -1 ]  # get last value of the equity column in USD
    pct_ann_return_svc = (ann_return_svc - 100000) / 100000 * 100

    print( "Results using Technical Indicators and SVC" )
    print( "Final portfolio value: {0:0.2f} USD".format( ann_return ) )
    print( "Total return: {0:0.2f} %".format( pct_ann_return ) )
    # print("Sharpe Ratio: {}".format(sharpe))
    print( "Total number of trades performed: {}".format( n_trades ) )
    print()

    print( "Results using Technical Indicators" )
    print( "Final portfolio value: {0:0.2f} USD".format( ann_return_TI ) )
    print( "Total return: {0:0.2f} %".format( pct_ann_return_TI ) )
    # print("Sharpe Ratio: {}".format(sharpe))
    print( "Total number of trades performed: {}".format( n_trades_TI ) )
    print()

    print( "Results using SVC" )
    print( "Final portfolio value: {0:0.2f} USD".format( ann_return_svc ) )
    print( "Total return: {0:0.2f} %".format( pct_ann_return_svc ) )
    # print("Sharpe Ratio: {}".format(sharpe))
    print( "Total number of trades performed: {}".format( n_trades_svc ) )
    print()

    # plotting performance and prices
    f, (ax1, ax2, ax3, ax4) = plt.subplots( 4, 1, sharex=True,
                                            figsize=(30, 20) )  # figure with 2 subplots and shared x axis
    f.subplots_adjust( hspace=0 )  # remove space between subplots
    x1 = data[ [ 'Close' ] ]  # .resample('W' ).mean()  # plot EUR/USD price resampled weekly
    ax1.plot( x1 )
    # ax1.set_title( 'System Performance', fontsize=23 )
    ax1.set_ylabel( 'EUR / USD Price', fontsize=23 )
    ax1.grid()
    for index, row in order_book.iterrows():
        trade = row[ 'Trade' ]
        if trade == 1:  # try using ax1.vlines
            ax1.vlines( x=index, y=order_book.loc[ index, 'Close' ], color='green' )
        elif trade == -1:
            ax1.vlines( x=index, y=order_book.loc[ index, 'Close' ], color='red' )

    x2 = performance[ [ 'Equity' ] ]
    ax2.set_ylabel( 'System Returns in $', fontsize=23 )
    ax2.plot( x2 )

    x3 = performance_TI[ [ 'Equity_TI' ] ]
    ax3.set_ylabel( 'Returns in $ using TI', fontsize=23 )
    ax3.plot( x3 )

    x4 = performance_svc[ [ 'Equity_svc' ] ]
    ax4.set_ylabel( 'Returns in $ using SVC', fontsize=23 )
    ax4.set_xlabel( 'Time', fontsize=25 )
    ax4.plot( x4 )

    plt.show()

    return data, order_book, order_book_TI, order_book_svc, equity_amount, equity_amount_TI, equity_amount_svc


# load datasets--------------------------------------------------------------------------------------------------------
EU = pd.read_csv( 'data/EURUSD_1H.csv', index_col=[ 'Time' ], usecols=[ 'Time', 'Close', 'Volume' ], parse_dates=True,
                  infer_datetime_format=True, dayfirst=True )
EUR_USD = EU.copy()

# load cryptocurrency dataFrame from poloniex exchange
# 1420070400 unix timestamp for 1st Jan 2015
# 1451606400 unix timestamp for 1st Jan 2016
usdt_eth2h = get_crypto_data( symbol='USDT_ETH', frequency=7200, start=1451606400, tocsv=False )[
    [ 'close', 'volume' ] ]
usdt_eth2h.shape
usdt_eth2h.rename( columns={ 'close': 'Close', 'volume': 'Volume' }, inplace=True )  # change names of columns

usdt_eth30mim = get_crypto_data( symbol='USDT_ETH', frequency=1800, start=1451606400, tocsv=False )[
    [ 'close', 'volume' ] ]
usdt_eth30min.shape
usdt_eth30min.rename( columns={ 'close': 'Close', 'volume': 'Volume' }, inplace=True )  # change names of columns
# ---------------------------------------------------------------------------------------------------------------------

# use_prices=True to use 100 price sequences, False - to use Technical indicators
X_train, X_test, y_train, y_test, X_val, y_val = get_data_for_ml( df=usdt_eth2h, use_prices=False,
                                                                  start_testing='2016-07-01', end_testing='2017-09-01',
                                                                  validation_start='2017-09-02', validation_end=None )

# use_prices=True to use 100 price sequences, False - to use Technical indicators
X_train, X_test, y_train, y_test, X_val, y_val = get_data_for_ml( df=EUR_USD, use_prices=False,
                                                                  start_testing='2015-06-01', end_testing='2017-06-01',
                                                                  validation_start='2017-06-02', validation_end=None )

# specify parameters to try for classifier
svc_parameters = [
    { 'kernel': [ 'poly' ], 'degree': [ 2, 3, 4 ], 'C': [ 1, 10, 100, 1000 ], 'gamma': [ 0.001, 0.01, 0.1 ] },
    {'kernel': ['rbf'], 'C': [1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1]},
    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]} ]  # DON'T USE gamma=1 in poly kernel and 0.0001

# do exhaustive search for best parameters to find those that max accuracy score and do K-fold cv with K=5
# manually input class weights based on the output from get_data_for_ml function
clf = GridSearchCV( SVC( class_weight={ -1: 2.60260973, 0: 0.44579905, 1: 2.68379205 }, cache_size=400 ),
                    param_grid=svc_parameters, cv=5, scoring='accuracy', n_jobs=-1, refit=True,
                    return_train_score=False, verbose=42 )

# fit input (X_train: price sequences / TIs) to target (y_train: 1, 0, -1)
clf.fit( X_train, y_train )

# print training results
print( "RESULTS on TRAINING dataset" )
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

# predict testing dataset and print results
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print( "RESULTS on TESTING dataset" )
print()
print('Support Vector Machine Classifier\n {}\n'.format(classification_report( y_test, y_pred,
                                                                               target_names=[ 'NegativeChange',
                                                                                              'SmallChange',
                                                                                              'PositiveChange' ] ) ) )
print("Accuracy Score: {0:0.2f} %".format(acc * 100))
MSE = mean_squared_error( y_test, y_pred )
print( "Mean Squared Error (MSE): {}".format( MSE ) )
print( "Root Mean Square Error (RMSE): {}".format( math.sqrt( MSE ) ) )
print( "Best score achieved by classifier: {}".format( clf.best_score_ ) )
print()
print( "Best Estimator: " )
print(clf.best_estimator_)

# compress is used to put all the files into a single pickle file
joblib.dump( clf.best_estimator_, 'eur_usd_2y_1y.pkl', compress=1 )  # save model trained on TI to pkl file
clf2 = joblib.load( 'svc_prices_mType.pkl' )  # load classifier to a variable


# predict validation set
y_pred2 = clf.predict( X_val )
acc2 = accuracy_score(y_val, y_pred2)
print('Support Vector Machine Classifier\n {}\n'.format(classification_report( y_val, y_pred2,
                                                                               target_names=[ 'NegativeChange',
                                                                                              'SmallChange',
                                                                                              'PositiveChange' ] ) ) )
print("Accuracy Score: {0:0.2f} %".format(acc2 * 100))
MSE_val = mean_squared_error( y_val, y_pred2 )
print( "Mean Squared Error (MSE): {}".format( MSE_val ) )
print( "Root Mean Square Error (RMSE): {}".format( math.sqrt( MSE_val ) ) )


# do trading__________________________________________________________________________________________
df_for_trading = make_trading_rules( EUR_USD )  # create dataset for trading

backtesting_data = df_for_trading[ '2017-06-02': ]  # leave only needed time frame
len( y_pred2 )
backtesting_data.shape
backtesting_data.drop( backtesting_data.tail( 1 ).index, inplace=True )  # drop last n rows
backtesting_data[ 'm_pred' ] = y_pred2  # attach market direction predictions from ml

backtested_df, new_order_book, new_order_book_TI, new_order_book_svc, eq, eq_TI, eq_svc = do_backtest(
    backtesting_data )

"""
models to try:

EUR-USD (1h)
1. Tech Indicators -- done
2. Price Sequences

ETH-USD
1. Tech Indicators (2h)
2. Price Sequences (2h)
3. Tech Indicators (30 min)
4. Price Sequences (30 min)

BTC-USD
1. Tech Indicators (2h)
2. Price Sequences (2h)
3. Tech Indicators (30 min)
4. Price Sequences (30 min)

"""
