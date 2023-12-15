import numpy as np
import pandas as pd

# 设置用户数、项目数、每个用户对应的项目数
num_users = 100
num_items = 150
num_items_per_user = 120

# 生成评分矩阵
ratings = np.random.randint(1, 6, size=(num_users, num_items_per_user))

# 生成用户id和项目id
user_ids = np.repeat(np.arange(num_users), num_items_per_user)
item_ids = np.tile(np.arange(num_items), num_users)[:num_users*num_items_per_user]

# 将评分矩阵展开为一维数组
ratings_flat = ratings.reshape(-1)

# 将用户id、项目id和对应的评分放到一个数据框中
df = pd.DataFrame({'user_id': user_ids, 'item_id': item_ids, 'rating': ratings_flat})

# 将数据保存到csv文件中
df.to_csv('ratings.csv', index=False)
