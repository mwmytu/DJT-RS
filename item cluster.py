import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def get_A():
	# 读取数据
	data = pd.read_csv('Data/recommendation.csv')
	# 选择需要聚类的字段
	X = data[
		['user_id', 'item_id', 'cat_id', 'seller_id', 'brand_id', 'time_stamp', 'action_type', 'age_range', 'gender']]
	# 将数据转换为数组
	X = X.values
	# 创建KMeans模型
	kmeans = KMeans(n_clusters=50, random_state=0).fit(X)
	# 聚类结果
	labels = kmeans.labels_
	# 打印聚类结果
	a = {}
	for i in range(10):
		a[i] = np.unique(data[labels == i]['item_id'].values)
	A = {
		'a1': a[0],
		'a2': a[1],
		'a3': a[2],
		'a4': a[3],
		'a5': a[4],
		'a6': a[5],
		'a7': a[6],
		'a8': a[7],
		'a9': a[8],
		'a10': a[9]
	}
	return A
