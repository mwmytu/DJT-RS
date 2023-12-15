import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score


df_ratings = pd.read_csv('../ml-100k/u.data')
df_movies = pd.read_csv('../ml-100k/u.item')


df = pd.merge(df_ratings, df_movies, on='movieId')


train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)


user_features = np.random.rand(train_data['userId'].nunique(), 10)  # 假设用户特征向量是10维
movie_features = np.random.rand(df_movies['movieId'].nunique(), 20)  # 假设电影特征向量是20维


m = np.random.rand(10, 20)  # 假设m是10行20列的矩阵
b = np.random.rand(10, 1)   # 假设b是10行1列的矩阵


recommendations = []
labels = []


for _, row in train_data.iterrows():
    user_id = row['userId']
    movie_id = row['movieId']
    rating = row['rating']


    user_feature = user_features[user_id - 1]
    movie_feature = movie_features[movie_id - 1]


    arm_features = np.dot(m, movie_feature) + b
    probabilities = arm_features / np.sum(arm_features)


    action = np.argmax(probabilities)
    recommendations.append(action)


    labels.append(rating)


    m[action] += np.outer(user_feature, movie_feature)
    b[action] += rating * user_feature


accuracy = accuracy_score(labels, recommendations)
recall = recall_score(labels, recommendations, average='weighted')
f1 = f1_score(labels, recommendations, average='weighted')

print("准确率: {:.4f}".format(accuracy))
print("召回率: {:.4f}".format(recall))
print("F1得分: {:.4f}".format(f1))
