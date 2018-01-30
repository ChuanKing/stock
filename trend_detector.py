import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt

max_width = 100000
max_height = 10000
pd.set_option('display.height', max_height)
pd.set_option("display.max_rows",max_height)
pd.set_option('display.max_columns', max_width)
pd.set_option('display.width', max_width)

################## Retrieve Stock Data ##################
# import datetime
# start =  datetime.datetime(2003, 1, 1)
# end = datetime.date.today()
# apple = web.DataReader("AAPL", "yahoo", start, end)
# apple.to_csv('apple.csv')

################## Import Stock Data ##################
stock = pd.read_csv('/Users/James/workspace/python/test/panda-test/apple.csv', index_col=0)
# stock = pd.DataFrame({
#     "Heigh": [9, 7, 6, 5],
#     "Low": [6, 3, 4, 3],
# })

################## 计算点 ##################
# 1. 处理包含关系
new_heigh=[]
new_low=[]

pre_index = None
moving_direction = "down"
for index, row in stock.iterrows():
    cur_heigh = row["High"]
    cur_low = row["Low"]
    
    if len(new_heigh) > 0:
        pre_heigh = new_heigh[-1]
        pre_low = new_low[-1]

        if pre_heigh >= cur_heigh and pre_low <= cur_low:
            if moving_direction == "up": cur_heigh = pre_heigh
            if moving_direction == "down": cur_low = pre_low
            new_heigh[-1] = np.nan
            new_low[-1] = np.nan

        elif pre_heigh <= cur_heigh and pre_low >= cur_low:
            if moving_direction == "up": cur_low = pre_low
            if moving_direction == "down": cur_heigh = pre_heigh
            new_heigh[-1] = np.nan
            new_low[-1] = np.nan

        else:
            moving_direction = "up" if cur_heigh > pre_heigh else "down"

    new_heigh.append(cur_heigh)
    new_low.append(cur_low)

stock["new_heigh"] = new_heigh
stock["new_low"] = new_low
print(stock)
# stock["Close"].plot()

# plt.show()