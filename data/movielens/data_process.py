import pandas as pd
import numpy as np
data = pd.read_csv("/ml-1m/ratings.dat", sep="::", names=["userID", "itemID", "ratings", "timestamp"])
user_num = max(data['userID'].unique())
item_num = max(data['itemID'].unique())

processed_data = data.iloc[:, :3]
processed_data['ratings'] = processed_data['ratings'] > 3

# 输出训练集和验证集8:2
rate = 0.8
shuffle_index = np.arange(len(processed_data))
np.random.shuffle(shuffle_index)
train_num = int(len(processed_data) * rate)

train_index = shuffle_index[:train_num]
valid_index = shuffle_index[train_num:]

train_data = processed_data.iloc[train_index]
valid_data = processed_data.iloc[valid_index]

# 保存
import json
train_data.to_csv("train.csv", index=False, header=False)
valid_data.to_csv("valid.csv", index=False, header=False)

info = {"user_num": str(user_num), "item_num": str(item_num)}
f = open("info.json", 'w')
json.dump(info, f)
f.close()