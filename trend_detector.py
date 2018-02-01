import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt

max_width = 100000
max_height = 10000
pd.set_option("display.height", max_height)
pd.set_option("display.max_rows",max_height)
pd.set_option("display.max_columns", max_width)
pd.set_option("display.width", max_width)

################## Retrieve Stock Data ##################
# import datetime
# start =  datetime.datetime(2003, 1, 1)
# end = datetime.date.today()
# apple = web.DataReader("AAPL", "yahoo", start, end)
# apple.to_csv("apple.csv")

################## Import Stock Data ##################
stock = pd.read_csv("/Users/James/workspace/python/test/panda-test/apple.csv", index_col=0)
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
stock["bottom"] = np.where(stock["new_low_min"] == stock["new_low"], -stock["new_low"], np.nan)

fractal = stock["top"].fillna(stock["bottom"]).values.tolist()
processing_type = "top" if stock["top"].first_valid_index() < stock["bottom"].first_valid_index() else "bottom"
pre_index = None

for i in range(len(fractal)): 
    if np.isnan(fractal[i]): continue

    if fractal[i] > 0:
        if pre_index is None:
            processing_type = "bottom"
            pre_index = i
        # 去掉太近的顶分型
        elif processing_type == "top" and i - pre_index < 4: 
            fractal[i] = np.nan
        # 现在的顶分型比之前的高，故去掉之前的顶分型
        elif processing_type == "bottom" and fractal[i] > fractal[pre_index]: 
            fractal[pre_index] = np.nan
            pre_index = i
        elif processing_type == "bottom": 
            fractal[i] = np.nan
        else:
            processing_type = "bottom"
            pre_index = i
    
    elif fractal[i] < 0:
        if pre_index is None:
            processing_type = "top"
            pre_index = i
        # 去掉太近的底分型
        if processing_type == "bottom" and i - pre_index < 4: 
            fractal[i] = np.nan
        # 现在的底分型比之前的低，故去掉之前的底分型
        elif processing_type == "top" and abs(fractal[i]) < abs(fractal[pre_index]): 
            fractal[pre_index] = np.nan
            pre_index = i
        elif processing_type == "top": 
            fractal[i] = np.nan
        else:
            processing_type = "top"
            pre_index = i

stock["fractal"] = fractal

################## 绘制线 ##################
stock["stroke"] = stock["fractal"].abs().interpolate()

################## 绘制线段 ##################
fractals = [[index, row["fractal"]] for index, row in stock[np.isfinite(stock["fractal"])].iterrows()]

# 1. 上升序列
up_features = []
up_features_fractals = fractals if fractals[0][1] > 0 else fractals[1:]
up_features_fractals = up_features_fractals if len(up_features_fractals) % 2 == 0 else up_features_fractals[:-1]

for i in range(len(up_features_fractals)):
    if i % 2 == 0:
        feature_top = up_features_fractals[i]
        feature_bottom = up_features_fractals[i + 1]
        up_features.append([feature_top[0], abs(feature_top[1]), abs(feature_bottom[1])])

# 2. 下降序列
down_features = []
down_features_fractals = fractals if fractals[0][1] < 0 else fractals[1:]
down_features_fractals = down_features_fractals if len(down_features_fractals) % 2 == 0 else down_features_fractals[:-1]

for i in range(len(down_features_fractals)):
    if i % 2 == 0:
        feature_bottom = down_features_fractals[i]
        feature_top = down_features_fractals[i + 1]
        down_features.append([feature_bottom[0], abs(feature_bottom[1]), abs(feature_top[1])])

# 3. 处理包含关系
up_features_clean = [up_features[0]]
for i in range(1, len(up_features)):
    pre_date = up_features_clean[-1][0]
    pre_high = up_features_clean[-1][1]
    pre_low = up_features_clean[-1][2]
    cur_date = up_features[i][0]
    cur_high = up_features[i][1]
    cur_low = up_features[i][2]

    if pre_high >= cur_high and pre_low <= cur_low:
        up_features_clean[-1] = [pre_date, pre_high, cur_low]
    
    elif pre_high < cur_high and pre_low > cur_low:
        up_features_clean[-1] = [cur_date, cur_high, pre_low]
    
    else:
        up_features_clean.append([cur_date, cur_high, cur_low])

down_features_clean = [down_features[0]]
for i in range(1, len(down_features)):
    pre_date = down_features_clean[-1][0]
    pre_low = down_features_clean[-1][1]
    pre_high = down_features_clean[-1][2]
    cur_date = down_features[i][0]
    cur_low = down_features[i][1]
    cur_high = down_features[i][2]

    if pre_high >= cur_high and pre_low <= cur_low:
        down_features_clean[-1] = [pre_date, pre_low, cur_high]
    
    elif pre_high < cur_high and pre_low > cur_low:
        down_features_clean[-1] = [cur_date, cur_low, pre_high]
    
    else:
        down_features_clean.append([cur_date, cur_low, cur_high])

stock["test1"] = np.nan
stock["test2"] = np.nan
for point in up_features_clean:
    stock.at[point[0], "test1"] = stock.loc[point[0]]["fractal"]

for point in down_features_clean:
    stock.at[point[0], "test2"] = stock.loc[point[0]]["fractal"]

# 4. 特征序列顶底分型
up_features_top = []
for i in range(1, len(up_features_clean) - 1):
    pre_high = up_features_clean[i-1][1]
    cur_high = up_features_clean[i][1]
    next_high = up_features_clean[i+1][1]
    
    if cur_high > pre_high and cur_high > next_high:
        up_features_top.append(up_features_clean[i][0])

down_features_bottom = []
for i in range(1, len(down_features_clean) - 1):
    pre_bottom = down_features_clean[i-1][1]
    cur_bottom = down_features_clean[i][1]
    next_bottom = down_features_clean[i+1][1]
    
    if cur_bottom < pre_bottom and cur_bottom < next_bottom:
        down_features_bottom.append(down_features_clean[i][0])

# 5. 标记线段点
stock["line-point"] = np.nan

for top in up_features_top:
    stock.at[top, "line-point"] = stock.loc[top]["fractal"]

for bottom in down_features_bottom:
    stock.at[bottom, "line-point"] = stock.loc[bottom]["fractal"]

# 6. 画线
stock["line"] = stock["line-point"].abs().interpolate()
################## 清理垃圾 ##################
stock.drop([
    "new_high_max",
    "new_low_min", 
    "new_high", 
    "new_low", 
    "top",
    "bottom"
], axis=1, inplace=True)


################## 输出 ##################
print(stock)
stock["stroke"].abs().plot()
stock["test1"].abs().interpolate().plot()
stock["test2"].abs().interpolate().plot()
# stock["Close"].plot()

plt.show()