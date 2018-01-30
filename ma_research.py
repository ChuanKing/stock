import numpy as np
import pandas as pd
import pandas_datareader.data as web

max_width = 100000
max_height = 20
pd.set_option('display.height', max_height)
pd.set_option("display.max_rows",max_height)
pd.set_option('display.max_columns', max_width)
pd.set_option('display.width', max_width)

################## Import Stock Data ##################
# start =  datetime.datetime(2010, 1, 1)
# end = datetime.date.today()
# apple = web.DataReader("AAPL", "yahoo", start, end)
# apple.to_csv('apple.csv')
stock = pd.read_csv('/Users/ggo935/James/workspace/python/test/panda-test/apple.csv', index_col=0)

################## Labeling Data ##################
holding_window = 10
trigger_threshold = 0.05
stock["next_" + str(holding_window) + "d_max"] = stock["Close"].rolling(window = holding_window, center = False).max().shift(1 - holding_window)
stock["next_" + str(holding_window) + "d_min"] = stock["Close"].rolling(window = holding_window, center = False).min().shift(1 - holding_window)
stock["Close_up_20"] = stock["Close"] * (1 + trigger_threshold)
stock["Close_down_20"] = stock["Close"] * (1 - trigger_threshold)
stock["Buy"] = np.where((stock["Close"] * (1 + trigger_threshold) < stock["next_" + str(holding_window) + "d_max"]) & (stock["Close"] * (1 - trigger_threshold) < stock["next_" + str(holding_window) + "d_min"]), 1, 0) 
stock["Sell"] = np.where(stock["Close"] * (1 - trigger_threshold) > stock["next_" + str(holding_window) + "d_min"], 1, 0) 
# print(stock.loc[stock["Buy"] == 1][["Close", "next_" + str(holding_window) + "d_max", "Close_up_20", "next_" + str(holding_window) + "d_min", "Close_down_20", "Buy", "Sell"]])

# ################## Calculate Moving Average ##################
for day in range(10, 200, 10):
    stock[str(day) + "_d"] = stock["Close"].rolling(window = day, center = False).mean()
stock.dropna(inplace=True)

################## Calculate Stock Price V.S. Moving Average ##################
for day in range(10, 200, 10):
    stock["up_" + str(day) + "d"] = np.where(stock["Close"] - stock[str(day) + "_d"] > 0, 1, 0)
    stock["down_" + str(day) + "d"] = np.where(stock["Close"] - stock[str(day) + "_d"] < 0, 1, 0)


################## Machine Learning ##################
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

X = stock.loc[:, ["up_10d", "up_20d", "up_30d", "up_40d", "up_50d", "up_60d"]]
y = stock["Buy"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

clf = MultinomialNB()
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

print(y_predict)

"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

best = ""
best_error = np.inf

for day in range(10, 200, 10):
    X = stock.loc[:, ["up_" + str(day) + "d"]]
    y = stock["Buy"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

    clf = LinearRegression()
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)

    error = mean_squared_error(y_test, y_predict)
    if error < best_error: 
        best = "up_" + str(day) + "d"
        best_error = error
    print(str(day) + "days MA error: ", error)
    print(y_predict)
print(best)





"""