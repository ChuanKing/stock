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
import datetime
ticker = "SPY"
start =  datetime.datetime(2013, 1, 1)
end = datetime.date.today()
stock = web.DataReader(ticker, "yahoo", start, end)
# stock.to_csv(ticker + ".csv")

################## Import Stock Data ##################
# stock = pd.read_csv("/Users/ggo935/James/workspace/python/stock/apple.csv", index_col=0)
# stock.index = pd.to_datetime(stock.index)
stock["Date"] = stock.index

################## 处理包含关系 ##################
data = [[index, row["High"], row["Low"]] for index, row in stock.iterrows()]
data_clean = [data[0]]
trend = "up"

for i in range(1, len(data)):
    pre_date, pre_high, pre_low = data_clean[-1]
    cur_date, cur_high, cur_low = data[i]

    if pre_high >= cur_high and pre_low <= cur_low:
        if trend == "up": data_clean[-1] = [pre_date, pre_high, cur_low]
        else: data_clean[-1] = [pre_date, cur_high, pre_low]
    
    elif pre_high < cur_high and pre_low > cur_low:
        if trend == "up": data_clean[-1] = [cur_date, cur_high, pre_low]
        else: data_clean[-1] = [cur_date, pre_high, cur_low]
    
    else:
        data_clean.append([cur_date, cur_high, cur_low])
        trend = "up" if data_clean[-1][1] > data_clean[-2][1] else "down"

# stock["High_clean"] = np.nan
# stock["Low_clean"] = np.nan

# for d in data_clean:
#     stock.at[d[0], "High_clean"] = d[1]
#     stock.at[d[0], "Low_clean"] = d[2]

################## 寻找分型 ##################
top = []
bottom = []
pre_index = None
processing_type = "top" if data_clean[0][1] < data_clean[1][1] else "bottom"

for i in range(1, len(data_clean) - 1):
    pre_date, pre_high, pre_low = data_clean[i-1]
    cur_date, cur_high, cur_low = data_clean[i]
    next_date, next_high, next_low = data_clean[i+1]

    if cur_high > pre_high and cur_high > next_high:
        # 1. 起始的顶分型
        if pre_index is None: 
            top.append(data_clean[i])
            pre_index, processing_type = i, "bottom"
        # 2. 正常的顶分型
        elif processing_type == "top" and i - pre_index >= 4:
            top.append(data_clean[i])
            pre_index, processing_type = i, "bottom"
        # 3. 在寻找底分型的过程中出现了顶分型，而且高于之前的顶分型
        if processing_type == "bottom" and cur_high > top[-1][1]: 
            top[-1] = data_clean[i]
            pre_index, processing_type = i, "bottom"
    
    if cur_low < pre_low and cur_low < next_low:
        # 1. 起始的底分型
        if pre_index is None: 
            bottom.append(data_clean[i])
            pre_index, processing_type = i, "top"
        # 2. 正常的底分型
        elif processing_type == "bottom" and i - pre_index >= 4:
            bottom.append(data_clean[i])
            pre_index, processing_type = i, "top"
        # 3. 在寻找底分型的过程中出现了底分型，而且高于之前的底分型
        if processing_type == "top" and cur_low < bottom[-1][2]: 
            bottom[-1] = data_clean[i]
            pre_index, processing_type = i, "top"

stock["fractals"] = np.nan

for d in top:
    stock.at[d[0], "fractals"] = d[1]
for d in bottom:
    stock.at[d[0], "fractals"] = d[2]

stock["stroke"] = stock["fractals"].interpolate(method="linear")

################## 寻找线段端点 ##################
# fractals = stock["fractals"].dropna().values.tolist()
fractals = stock[["Date", "fractals"]].dropna().values.tolist()

# 1. 寻找特征序列
up_sequence = []
down_sequence = []

for i in range(len(fractals) - 1):
    # 1. 向下笔为上升序列特征值
    if fractals[i][1] > fractals[i+1][1]:
        up_sequence.append([fractals[i][0], fractals[i][1], fractals[i+1][1]])
    # 2. 向上笔为下降序列特征值
    if fractals[i][1] < fractals[i+1][1]:
        down_sequence.append([fractals[i][0], fractals[i+1][1], fractals[i][1]])

# 2. 处理包含关系
trend = "up"
up_sequence_clean = [up_sequence[0]]
for i in range(1, len(up_sequence)):
    pre_date, pre_high, pre_low = up_sequence_clean[-1]
    cur_date, cur_high, cur_low = up_sequence[i]

    if pre_high >= cur_high and pre_low <= cur_low:
        if trend == "up": up_sequence_clean[-1] = [pre_date, pre_high, cur_low]
        else: up_sequence_clean[-1] = [pre_date, cur_high, pre_low]
    
    elif pre_high < cur_high and pre_low > cur_low:
        if trend == "up": up_sequence_clean[-1] = [cur_date, cur_high, pre_low]
        else: up_sequence_clean[-1] = [cur_date, pre_high, cur_low]
    
    else:
        up_sequence_clean.append([cur_date, cur_high, cur_low])
        trend = "up" if up_sequence_clean[-1][1] > up_sequence_clean[-2][1] else "down"

trend = "up"
down_sequence_clean = [down_sequence[0]]
for i in range(1, len(down_sequence)):
    pre_date, pre_high, pre_low = down_sequence_clean[-1]
    cur_date, cur_high, cur_low = down_sequence[i]

    if pre_high >= cur_high and pre_low <= cur_low:
        if trend == "up": down_sequence_clean[-1] = [pre_date, pre_high, cur_low]
        else: down_sequence_clean[-1] = [pre_date, cur_high, pre_low]
    
    elif pre_high < cur_high and pre_low > cur_low:
        if trend == "up": down_sequence_clean[-1] = [cur_date, cur_high, pre_low]
        else: down_sequence_clean[-1] = [cur_date, pre_high, cur_low]
    
    else:
        down_sequence_clean.append([cur_date, cur_high, cur_low])
        trend = "up" if down_sequence_clean[-1][1] > down_sequence_clean[-2][1] else "down"

# 3. 寻找特征序列顶底分型
up_sequence_top = []
for i in range(1, len(up_sequence_clean) - 1):
    pre_date, pre_high, pre_low = up_sequence_clean[i-1]
    cur_date, cur_high, cur_low = up_sequence_clean[i]
    next_date, next_high, next_low = up_sequence_clean[i+1]

    if cur_high > pre_high and cur_high > next_high:
        up_sequence_top.append(up_sequence_clean[i])

down_sequence_bottom = []
for i in range(1, len(down_sequence_clean) - 1):
    pre_date, pre_high, pre_low = down_sequence_clean[i-1]
    cur_date, cur_high, cur_low = down_sequence_clean[i]
    next_date, next_high, next_low = down_sequence_clean[i+1]

    if cur_low < pre_low and cur_low < next_low:
        down_sequence_bottom.append(down_sequence_clean[i])

# 4. 准备数据
sequence_date_clean = sorted([point[0] for point in up_sequence_clean + down_sequence_clean])
sequence_value_clean = dict()

for point in up_sequence_clean: sequence_value_clean[point[0]] = point[1]
for point in down_sequence_clean: sequence_value_clean[point[0]] = point[2]

up_sequence_top = [point[0] for point in up_sequence_top]
down_sequence_bottom = [point[0] for point in down_sequence_bottom]

# 5. 选取线段点
first_line_point = sorted(up_sequence_top + down_sequence_bottom)[0]
processing_type = "top" if first_line_point in up_sequence_top else "bottom"
pre_index = sequence_date_clean.index(first_line_point)

line = [first_line_point]
for i in range(pre_index + 1, len(sequence_date_clean)):
    pre_date = line[-1]
    cur_date = sequence_date_clean[i]

    if cur_date in up_sequence_top:
        if pre_date in up_sequence_top and sequence_value_clean[cur_date] > sequence_value_clean[pre_date]:
            line[-1] = cur_date
            pre_index = i
        elif pre_date in down_sequence_bottom and i - pre_index >= 3:
            line.append(cur_date)
            pre_index = i

    elif cur_date in down_sequence_bottom:
        if pre_date in down_sequence_bottom and sequence_value_clean[cur_date] < sequence_value_clean[pre_date]:
            line[-1] = cur_date
            pre_index = i
        elif pre_date in up_sequence_top and i - pre_index >= 3:
            line.append(cur_date)
            pre_index = i

# 6. 处理第一点和最后一点
high = stock["High"].values.tolist()
date = stock["Date"].values.tolist()

first_high = high[0]
last_high = high[-1]

first_date = date[0]
last_date = date[-1]

if (first_high - sequence_value_clean[line[0]]) * (sequence_value_clean[line[0]] - sequence_value_clean[line[1]]) > 0:
    line = line[1:]

if (sequence_value_clean[line[-2]] - sequence_value_clean[line[-1]]) * (sequence_value_clean[line[-1]] - last_high) > 0:
    line = line[:-1]

line = [first_date] + line + [last_date]
sequence_value_clean[first_date] = first_high
sequence_value_clean[last_date] = last_high

# 7. 把线段点放进data framwork
stock["line-point"] = np.nan

for d in line:
    stock.at[d, "line-point"] = sequence_value_clean[d]

################## 绘制线段 ##################
line_point = stock["line-point"].values.tolist()
high = stock["High"].values.tolist()

line_point[0] = high[0]
line_point[-1] = high[-1]

stock["line"] = line_point
stock["line"] = stock["line"].interpolate()

################## 输出 ##################
print(stock[["High", "Low", "fractals", "stroke", "line-point", "line"]])

import toolbox
toolbox.pandas_candlestick_ohlc(stock, otherseries=['stroke', 'line'])
plt.show()


