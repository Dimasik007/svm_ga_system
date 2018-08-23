import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
import pandas as pd
import numpy as np
import talib as ta
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

sns.set()  # set beautiful style for graphs

data_path = '/Users/Dimasik007/Desktop/Programming/PyCharm/Dissertation/svm_ga_system/data/'

# load csv file to df
EU = pd.read_csv(data_path+'EURUSD_1H.csv', index_col=['Time'], usecols=['Time', 'Close', 'Volume'],
                 parse_dates=True, infer_datetime_format=True, dayfirst=True)

EUR_USD = EU.copy()


def get_TIs( df, dropnan=True ):
    """ calculate technical indicators with ta-lib
    requires a df with a column named "Close" """

    # to delete rows which have 0 volume / delete weekends
    df = df[ (df[ [ 'Volume' ] ] != 0).all( axis=1 ) ]

    df[ 'RSI' ] = ta.RSI( df.Close.values, timeperiod=14 )
    df[ 'ROC' ] = ta.ROC( df.Close.values, timeperiod=10 )
    df[ 'SMA' ] = ta.SMA( df.Close.values, timeperiod=30 )
    df[ 'EMA' ] = ta.EMA( df.Close.values, timeperiod=30 )
    df[ 'WMA' ] = ta.WMA( df.Close.values, timeperiod=30 )
    df[ 'MACD' ], df[ 'macdSignal' ], df[ 'macdHist' ] = ta.MACD( df.Close.values, fastperiod=12,
                                                                  slowperiod=26, signalperiod=9 )

    # to label different market types ( 1 - Uptrend, 0 - Sideways, -1 - Downtrend )
    df[ 'mType' ] = np.where( df[ 'ROC' ] > 0.5, 1, (np.where( df[ 'ROC' ] < -0.5, -1, 0 )) )

    if dropnan:
        df.dropna( inplace=True )

    return df


def trading_rules( df ):
    """ generate trading rules based on technical indicators: [RSI, ROC, SMA, EMA, WMA and MACD] """

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


EUR_USD = get_TIs(EUR_USD, True)
EUR_USD = trading_rules(EUR_USD)


# OPTIMISATION LAYER SVM_______________________________________________________________________________________
def normalise( x ):
    """ standardize features by removing the mean and scaling to unit variance """
    scaler = StandardScaler()
    x_norm = scaler.fit_transform( x.values )
    x_norm = pd.DataFrame( x_norm, index=x.index, columns=x.columns )

    return x_norm


def make_lags( df, max_lag, min_lag=0, separator='_' ):
    """ shift values in given columns of dataFrame """
    values = []
    for i in range( min_lag, max_lag + 1 ):
        values.append( df.shift( i ).copy() )
        values[ -1 ].columns = [ c + separator + str( i ) for c in df.columns ]
    return pd.concat( values, axis=1 )


def create_features_labels( df, price_seq=True, hm_lags=99 ):
    """
    prepare dataset fot machine learning
    if price_seq==True, return df with hm_lags price sequences and corresponding to price 0 market type
    else returns a dataFrame with sequences of technical indicators and market type shifted by 1
    """

    if price_seq:
        j = make_lags( df[[ 'Close' ] ], hm_lags )  # create df with 100 price sequences
        j[ 'mType' ] = df[ 'mType' ]  # label price sequences with market type
        j.dropna(inplace=True)  # drop NaN values left after shifting values
        return j
    else:
        df[ 'mType1' ] = df['mType'].shift(1).fillna(9).astype(int)  # create new column with shifted market type
        df = df[( df[[ 'mType1' ]] != 9).all(axis=1)]  # drop rows left after shifting market type
        df = df[[ 'RSI', 'ROC', 'SMA', 'EMA', 'WMA', 'MACD', 'macdSignal', 'macdHist', 'mType1' ]]
        return df


price_seq = create_features_labels(EUR_USD, price_seq=True)  # type 1
tis_seq = create_features_labels(EUR_USD, price_seq=False)  # type 2

# machine learning here________________________________________________________________________________________
# preparing data for ml
data_train = a[:'2005-12-31']  # create training df
y = data_train['mType1'].values  # target which we predict ( .values to transform it to ndarray )
# y = label_binarize(y, classes=[-1, 0, 1])  ######### probably not needed
X = data_train.drop(['mType', 'mType1', 'Close', 'Volume'], axis=1)  # data TI type 2
X = normalise(X)  # normalise X ( leads to x3 accuracy improvement )
X = X.values  # change type to ndarray
# X = data_train[['RSI', 'ROC', 'SMA', 'EMA', 'WMA', 'MACD', 'macdSignal', 'macdHist']]  # data TI sequences type 2

# validation set
data_val = a['2005-01-01':'2006-12-31']
y_val = data_val['mType1'].values  # target
X_val = data_val.drop(['mType', 'mType1', 'Close', 'Volume'], axis=1)  # use TI type 2
# X_val = data_val[['Open', 'High', 'Low', 'Close']]  # data price sequences type 1
X_val = normalise(X_val)
X_val = X_val.values
# TODO don't forget to normalise the data X_val
# X_val = data_val[['RSI', 'ROC', 'SMA', 'EMA', 'WMA', 'MACD', 'macdSignal', 'macdHist']]  # data TI sequences type 2

# splitting data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print("Training data: {} and target: {}".format(X_train.shape, y_train.shape))
print("Testing data: {} and target: {}".format(X_test.shape, y_test.shape))

# adjust weights of the classes to account for unbalanced data
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
print(class_weights)  # use that in class_weight parameter of the classifier, as it accepts only dict

# check the sum of each value
unique, counts = np.unique(y, return_counts=True)
dict(zip(unique, counts))

# specify parameters to try for classifier
svc_parameters = [{'kernel': ['poly'], 'degree': [2, 3, 4], 'C': [1, 10, 100, 1000],
                   'gamma': [0.0001, 0.001, 0.01, 0.1]},
                  {'kernel': ['rbf'], 'C': [1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1]},
                  {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
                  ]  # DON'T USE gamma=1 in poly kernel

# TODO try using ['accuracy', 'f1'] in scoring parameter
# do exhaustive search for best parameters to find those that max accuracy score and do K-fold with K=5
clf = GridSearchCV(SVC(class_weight={-1: 3.74030354, 0: 0.40926285, 1: 3.45752143}),
                   param_grid=svc_parameters, cv=5, scoring='accuracy',
                   n_jobs=-1, refit=True, return_train_score=False, verbose=3)

# fit input (X_train: price sequences / TIs) to target (y_train: 1, 0, -1)
clf.fit(X_train, y_train)

# print all the results
clf.score(X_train, y_train)
print("Best parameters set found on development set:")
print(clf.best_params_)
print()
print("Grid scores on training set:")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
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
print(clf.best_score_)
print(clf.best_estimator_)
# compress is used to put all the files into a single pickle file
joblib.dump(clf.best_estimator_, 'svc_ti.pkl', compress=1)  # save model trained on TI to pkl file
joblib.dump(clf.best_estimator_, 'svc_prices.pkl', compress=1)  # save model trained on price sequences to pkl file

clf2 = joblib.load('svc_ti.pkl')  # load classifier to a variable


#####################################################################################
count = 100
for i in range(0, count):
    b["Close_{}".format(str(i+1))] = b["Close"].shift(i+1)

y_pred2 = clf2.predict(X_val)
acc2 = accuracy_score(y_val, y_pred2)
print('Support Vector Machine Classifier\n {}\n'.format(classification_report(y_val, y_pred2,
                                                                              target_names=['Downtrend',
                                                                                            'Sideways',
                                                                                            'Uptrend'])))
print("Accuracy Score: {0:0.2f} %".format(acc2 * 100))


# TODO print evaluation for non binary labels
def scores(model, X, y):
    # print results
    # for model in models:
    y_pred = model.predict(X)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    print("Predictions: {}".format(y_pred[0:5]))
    print("Precision score: {}".format(prec))
    print("Recall score: {}".format(rec))
    print("Accuracy Score: {0:0.2f} %".format(acc * 100))
    print("F1 Score: {0:0.4f}".format(f1))


# rebalance data set
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

# print scores of clf.predict ( fix non binary labels )
scores(clf, X_test, y_test)

# plotting TIs and closing price on one plot
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20, 10))  # create figure with 2 subplots with shared x axis
f.subplots_adjust(hspace=0)  # remove space between subplots
x1 = EUR_USD['2017-01-01':'2018-06-01'][['Close']].resample('W').mean()  # plot EUR/USD price resampled weekly
ax1.plot(x1)
ax1.set_title('EUR/USD price and ROC indicator', fontsize=16)
ax1.set_ylabel('EUR/USD price', fontsize=12)

x2 = EUR_USD['2017-01-01':'2018-06-01'][['ROC']].resample('W').mean()
ax2.set_ylabel('9 ROC', fontsize=12)
ax2.set_xlabel('Year', fontsize=16)
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

# doesn't work now
# alternative implementation to speed up classification process by
# training several classifiers on different subsets of data
n_estimators = 10
score_eval = ['accuracy', 'f1']
clf3 = GridSearchCV(BaggingClassifier(SVC(class_weight={-1: 3.74030354, 0: 0.40926285, 1: 3.45752143}),
                                      max_samples=1.0 / n_estimators, n_estimators=n_estimators),
                    param_grid=svc_parameters, cv=5, scoring='accuracy',
                    n_jobs=-1, refit=True, return_train_score=False, verbose=3)

######
