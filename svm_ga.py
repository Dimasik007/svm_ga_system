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
from sklearn.externals import joblib
import pickle
sns.set()

'''
to do list:
FX data layer:
1) load csv EUR/USD + 
2) calculate Technical Indicators ( RSI, ROC, SMA, EMA, WMA, MACD ) + 
3) implement voting system based on TI 

Optimisation Layer
    SVM
* train SVM on past price data and on TI values and compare
1) implement SVM which classifies market state (1, 0, -1) + 
2) implement grid search algorithm to optimise parameters of SVM + 
3) implement K-fold cross validation method to assure model not biased + 
4) implement evaluation methods ( Precision, Recall, Accuracy ) + 
5) save model with pickle + 

    Backtesting
1) implement simple backtester
2) compare this model with a voting classifier
3) compare both models on cryptocurrency prices
'''
data_path = '/Users/Dimasik007/Desktop/Programming/PyCharm/Dissertation/svm_ga_system/data/'

# FX Data Layer

# load csv file to df
EU = pd.read_csv(data_path+'EURUSD_1H.csv', index_col=['Time'], usecols=['Time', 'Close', 'Volume'],
                 parse_dates=True, infer_datetime_format=True, dayfirst=True)

EUR_USD = EU.copy()

# to delete rows which have 0 volume / delete weekends
EUR_USD = EUR_USD[(EUR_USD[['Volume']] != 0).all(axis=1)]

# calculate technical indicators with ta-lib
EUR_USD['RSI'] = ta.RSI(EUR_USD.Close.values, timeperiod=14)
EUR_USD['ROC'] = ta.ROC(EUR_USD.Close.values, timeperiod=10)
EUR_USD['SMA'] = ta.SMA(EUR_USD.Close.values, timeperiod=30)
EUR_USD['EMA'] = ta.EMA(EUR_USD.Close.values, timeperiod=30)
EUR_USD['WMA'] = ta.WMA(EUR_USD.Close.values, timeperiod=30)
EUR_USD['MACD'], EUR_USD['macdSignal'], \
        EUR_USD['macdHist'] = ta.MACD(EUR_USD.Close.values,
                                      fastperiod=12, slowperiod=26, signalperiod=9)

# to label different market types ( 1 - Uptrend, 0 - Sideways, -1 - Downtrend )
EUR_USD['mType'] = np.where(EUR_USD['ROC'] > 0.5, 1,
                            (np.where(EUR_USD['ROC'] < -0.5, -1, 0)))

# shift positively - move values by x values down
EUR_USD['mType1'] = EUR_USD['mType'].shift(1).fillna(9).astype(int)
# delete rows with value 9 in shifted market type
EUR_USD = EUR_USD[(EUR_USD[['mType1']] != 9).all(axis=1)]

# drop nan values left from TI
EUR_USD.dropna(inplace=True)

# trading rules
EUR_USD['R_RSI'] = np.where(EUR_USD['RSI'] > 70, -1,
                            (np.where(EUR_USD['RSI'] < 30, 1, 0)))
EUR_USD['R_ROC'] = np.where(EUR_USD['ROC'] > 0, 1,
                            (np.where(EUR_USD['ROC'] < 0, -1, 0)))
EUR_USD['R_SMA'] = np.where(EUR_USD['SMA'] < EUR_USD['Close'], 1,
                            (np.where(EUR_USD['SMA'] > EUR_USD['Close'], -1, 0)))
EUR_USD['R_EMA'] = np.where(EUR_USD['EMA'] < EUR_USD['Close'], 1,
                            (np.where(EUR_USD['EMA'] > EUR_USD['Close'], -1, 0)))
EUR_USD['R_WMA'] = np.where(EUR_USD['WMA'] < EUR_USD['Close'], 1,
                            (np.where(EUR_USD['WMA'] > EUR_USD['Close'], -1, 0)))
EUR_USD['R_MACD'] = np.where(EUR_USD['MACD'] > 0, 1,
                             (np.where(EUR_USD['MACD'] < 0, -1, 0)))

col_list = ['R_RSI', 'R_ROC', 'R_SMA', 'R_EMA', 'R_WMA', 'R_MACD']
EUR_USD['Vote'] = EUR_USD[col_list].sum(axis=1) / len(col_list)
# TODO try different trading rules, eg buy/sell when ROC +/-0.15 etc.

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


# OPTIMISATION LAYER SVM______________________________________________________
def normalise(x):
    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    x_norm = scaler.fit_transform(x.values)
    x_norm = pd.DataFrame(x_norm, index=x.index, columns=x.columns)

    return x_norm


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


def table2lags(table, max_lag, min_lag=0, separator='_'):
    """ Given a dataframe, return a dataframe with different lags of all its columns """
    values = []
    for i in range(min_lag, max_lag + 1):
        values.append(table.shift(i).copy())
        values[-1].columns = [c + separator + str(i) for c in table.columns]
    return pd.concat(values, axis=1)


a = EUR_USD.copy()
j = table2lags(a[['Close']], 99)
print(j.head())
j['mType'] = a['mType']
j.dropna(inplace=True)


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

# TODO use BaggingClassifier to speed up training of classifier
# specify parameters for classifier
svc_parameters = [{'kernel': ['poly'], 'degree': [2, 3, 4], 'C': [1, 10, 100, 1000],
                   'gamma': [0.0001, 0.001, 0.01, 0.1]},
                  {'kernel': ['rbf'], 'C': [1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1]},
                  {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
                  ]  # DON'T USE gamma=1 in poly kernel

# do exhaustive search for best parameters to find those that max accuracy score and do K-fold with K=5
clf = GridSearchCV(SVC(class_weight={-1: 3.74030354, 0: 0.40926285, 1: 3.45752143}),
                   param_grid=svc_parameters, cv=5, scoring='accuracy',
                   n_jobs=-1, refit=True, return_train_score=False, verbose=3)
"""
# alternative implementation to speed up classification process by 
# training several classifiers on different subsets of data
n_estimators = 10
clf = GridSearchCV(BaggingClassifier(SVC(class_weight={-1: 3.74030354, 0: 0.40926285, 1: 3.45752143}),
                   param_grid=svc_parameters, cv=5, scoring='accuracy',
                   n_jobs=-1, refit=True, return_train_score=False, verbose=100), max_samples=1.0 / n_estimators,
                   n_estimators=n_estimators)
"""

# fit input (X_train: price sequences / TIs) to target (y_train: 1, 0, -1)
clf.fit(X_train, y_train)

# print all the results
clf.score(X_train, y_train)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on training set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

# print("{0:0.3f} (+/-{0:0.03f}) for {}".format(mean, std * 2, params))

#
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

# print scores of clf.predict ( fix non binary labels )
scores(clf, X_test, y_test)

"""
clf = OneVsRestClassifier(SVC(kernel='rbf', C=10, gamma=0.001))
clf.fit(X_train, y_train)
clf.score(X_train, y_train)
"""

# clf = SVC(class_weight={-1: 3.74237288, 0: 0.40858623, 1: 3.5047619})

"""
# get best parameters for classifier
for i in range(1):
    grid = GridSearchCV(param_grid[i], scoring='f1').fit(X_train, y_train)
    print(grid.best_params_)
    model_best = grid.best_estimator_
"""

# save model to pickle file
# saved_model = pickle.dumps(clf)
# clf2 = pickle.loads(saved_model)
# clf2.predict(X)

"""
with open('svc_model_prices.pickle', 'wb') as f:
    pickle.dump(clf, f)
pickle_in = open('svc_model_prices.pickle', 'rb')
clf_from_pickle = pickle.load(pickle_in)

# do cross validation
Kcv = KFold(n_splits=5)
prediction_cv = cross_val_score(clf, X, y, cv=cv)

# defining classification algorithm
clf = SVC()
# fit input (X_train: price sequences) to target (Y_train: 1, 0, -1)
clf.fit(X_train, y_train)


scores(clf, X_test, y_test)
"""

# score on validation set
scores(model_best, X_val, y_val)

count = 100
for i in range(0, count):
    b["Close_{}".format(str(i+1))] = b["Close"].shift(i+1)

'''
def series_to_supervised( data, n_in=1, n_out=1, dropnan=True ):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type( data ) is list else data.shape[ 1 ]
    df = pd.DataFrame( data )
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range( n_in, 0, -1 ):
        cols.append( df.shift( i ) )
        names += [ ('var%d(t-%d)' % (j + 1, i)) for j in range( n_vars ) ]
    # forecast sequence (t, t+1, ... t+n)
    for i in range( 0, n_out ):
        cols.append( df.shift( -i ) )
        if i == 0:
            names += [ ('var%d(t)' % (j + 1)) for j in range( n_vars ) ]
        else:
            names += [ ('var%d(t+%d)' % (j + 1, i)) for j in range( n_vars ) ]
    # put it all together
    agg = pd.concat( cols, axis=1 )
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna( inplace=True )
    return agg


data = series_to_supervised(EUR_USD[['RSI', 'ROC', 'SMA', 'EMA', 'WMA',
                                     'MACD', 'macdSignal', 'macdHist']].values, 2, 2, dropnan=False)
data.head()
'''
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

clf2 = joblib.load('svc_ti.pkl')
y_pred2 = clf2.predict(X_val)
acc2 = accuracy_score(y_val, y_pred2)
print('Support Vector Machine Classifier\n {}\n'.format(classification_report(y_val, y_pred2,
                                                                              target_names=['Downtrend',
                                                                                            'Sideways',
                                                                                            'Uptrend'])))
print("Accuracy Score: {0:0.2f} %".format(acc2 * 100))
