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

################## 处理包含关系 ##################
new_high=[]
new_low=[]

pre_index = None
moving_direction = "down"
for index, row in stock.iterrows():
    cur_high = row["High"]
    cur_low = row["Low"]
    
    if len(new_high) > 0:
        pre_high = new_high[-1]
        pre_low = new_low[-1]

        if pre_high >= cur_high and pre_low <= cur_low:
            if moving_direction == "up": cur_high = pre_high
            if moving_direction == "down": cur_low = pre_low
            new_high[-1] = np.nan
            new_low[-1] = np.nan

        elif pre_high <= cur_high and pre_low >= cur_low:
            if moving_direction == "up": cur_low = pre_low
            if moving_direction == "down": cur_high = pre_high
            new_high[-1] = np.nan
            new_low[-1] = np.nan

        else:
            moving_direction = "up" if cur_high > pre_high else "down"

    new_high.append(cur_high)
    new_low.append(cur_low)

stock["new_high"] = new_high
stock["new_low"] = new_low
stock["new_high"].interpolate(inplace=True)
stock["new_low"].interpolate(inplace=True)

################## 寻找分型 ##################
stock["new_high_max"] = stock["new_high"].rolling(window = 3, center = True).max()
stock["new_low_min"] = stock["new_low"].rolling(window = 3, center = True).min()
stock["top"] = np.where(stock["new_high_max"] == stock["new_high"], stock["new_high"], np.nan)
stock["bottom"] = np.where(stock["new_low_min"] == stock["new_low"], stock["new_low"], np.nan)

top = []
bottom = []
pre_index = None
processing_colume = "top" if stock["top"].first_valid_index() < stock["bottom"].first_valid_index() else "bottom"

for index, row in stock.iterrows():
    if np.isnan(row[processing_colume]):
        top.append(np.nan)
        bottom.append(np.nan)

    else:
        if pre_index is None or len(top) - pre_index >= 4:
            if processing_colume == "top":
                top.append(row[processing_colume])
                bottom.append(np.nan)
                processing_colume = "bottom"
                pre_index = len(top)
            else:
                bottom.append(row[processing_colume])
                top.append(np.nan)
                processing_colume = "top"
                pre_index = len(top)
        else:
            top.append(np.nan)
            bottom.append(np.nan)

stock["new_top"] = top
stock["new_bottom"] = bottom

stock["line-point"] = stock["top"].fillna(stock["bottom"])
stock["line"] = stock["line-point"].interpolate()

################## 绘制线 ##################
# stock["line-point"] = stock["top"].fillna(stock["bottom"])
# stock["line"] = stock["line-point"].interpolate()

# stock["trend"] = np.where(stock["line"] - stock["line"].shift(1) > 0, "up", np.nan)
# stock["trend"] = np.where(stock["line"] - stock["line"].shift(1) < 0, "down", stock["trend"])


################## 清理垃圾 ##################
stock.drop([
    "new_high_max",
    "new_low_min", 
    # "pre_new_high", 
    # "pre_new_low",
    # "new_high", 
    # "new_low", 
    # "next_new_high", 
    # "next_new_low",
    # "top",
    # "bottom"
], axis=1, inplace=True)


################## 输出 ##################
print(stock)
stock["line"].plot()

plt.show()