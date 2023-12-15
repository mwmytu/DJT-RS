import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# 读取数据
data = pd.read_csv('Data/ITEM.csv')

# 特征处理
# 对标识符字段进行独热编码
categorical_cols = ['user_id', 'item_id', 'cat_id', 'seller_id', 'brand_id']
encoder = OneHotEncoder(sparse=False)
encoded_features = encoder.fit_transform(data[categorical_cols])

# 时间特征提取
data['time_stamp'] = pd.to_datetime(data['time_stamp'])
data['year'] = data['time_stamp'].dt.year
data['month'] = data['time_stamp'].dt.month
data['day'] = data['time_stamp'].dt.day
# 其他时间特征的提取...


discrete_cols = ['action_type', 'age_range', 'gender']
for col in discrete_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])


continuous_cols = ['time_stamp']
scaler = MinMaxScaler()
data[continuous_cols] = scaler.fit_transform(data[continuous_cols])


data.to_csv('processed_data.csv', index=False)
