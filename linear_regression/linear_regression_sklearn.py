import pandas as pd
import quandl
import math
import numpy as np
import datetime
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

quandl.ApiConfig.api_key = "JRZ9Kje7vwVstSopjRdX"

# Get dataset from quandl
df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# We want to predict the closing stock prices
forecast_col = 'Adj. Close'
# fill the data where data is not present
df.fillna(-99999, inplace=True)

# number of days in advance
forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)

# label is the stock price after a couple of days
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
# print(df.head())

# For the features, we take all the columns except the label
# The features are: HL_PCT, PCT_CHANGE, Adj. Close, Adj. Volume
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
# predict the stock prices of the latest prices
X_lately = X[-forecast_out:]

y = np.array(df['label'])

# To standardize the dataset between -1 and 1
X = preprocessing.scale(X)

# X = X[:-forecast_out+1]
df.dropna(inplace=True)
y = np.array(df['label'])

# split the dataset into training and testing datasets
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)

# testing the model and getting the sum of square of the errors
accuracy = clf.score(x_test, y_test)

# if we want to use a different algorithm: clf = svm.SVR() for support vector machines

forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_set)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()