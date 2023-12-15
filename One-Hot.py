import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
# 打开CSV文件
data = pd.read_csv('u.csv')


encoder = OneHotEncoder()


users = data['user'].values.reshape(-1, 1)
items = data['item'].values.reshape(-1, 1)
ratings = data['rating'].values.reshape(-1, 1)


user_encoded = encoder.fit_transform(users).toarray()
item_encoded = encoder.fit_transform(items).toarray()
rating_encoded = encoder.fit_transform(ratings).toarray()


features = pd.DataFrame(
    data=np.concatenate([user_encoded, item_encoded, rating_encoded], axis=1).astype(float),  # 将数据类型转换为float
    columns=[f'user_{i}' for i in range(user_encoded.shape[1])] +
            [f'item_{i}' for i in range(item_encoded.shape[1])] +
            [f'rating_{i}' for i in range(rating_encoded.shape[1])]
)
features = features.astype(float)

data['features'] = features.values.tolist()
print(type(data['features']))

data.to_csv('uu.csv', index=False)
