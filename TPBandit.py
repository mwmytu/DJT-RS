import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


class TPBandit:

    def __init__(self, alpha=0.25, exploration_factor=2):
        self.alpha = alpha
        self.exploration_factor = exploration_factor
        self.articles = None
        self.user_params = {}
        self.interactions = set()

    def set_articles(self, articles):
        self.articles = articles
        for i, article in enumerate(articles):
            self.user_params[i] = {'A': np.identity(len(article)), 'b': np.zeros(len(article))}

    def update(self, user_id, item_id, reward):
        if reward == 1:
            item_id = tuple(item_id)
            self.interactions.add((user_id, item_id))
            if user_id in self.user_params:
                params = self.user_params[user_id]
                params['A'] += np.outer(item_id, item_id)
                params['b'] += reward * item_id

    def recommend(self, user_id):
        if user_id not in self.user_params:
            return None

        params = self.user_params[user_id]
        scores = []
        for article_id, article in enumerate(self.articles):
            A_inv = np.linalg.inv(params['A'])
            theta = np.dot(A_inv, params['b'])
            x = article.reshape(1, -1)  # Convert x to a 2D array
            exploration_term = self.exploration_factor * np.sqrt(np.log(len(self.articles)) / (np.diag(np.dot(np.dot(x, A_inv), x.T)) + 1e-6))
            score = np.dot(theta.T, x.T) + self.alpha * exploration_term
            scores.append(score)
        recommended_item_id = np.argmax(scores)
        recommended_item = self.articles[recommended_item_id]
        return recommended_item


# 读取物品特征向量数据集
item_features = pd.read_csv('DJT-RS/Data/Features.csv')

# 读取用户交互数据集
user_interactions = pd.read_csv('DCMAB/Data/ITEM.csv')

# 提取物品特征向量
items = item_features.drop('item_id', axis=1).values

users = user_interactions['user_id'].unique()

# 创建TPBandit对象
tp_bandit = TPBandit(alpha=0.25, exploration_factor=2)
tp_bandit.set_articles(items)

# 记录推荐成功和总推荐次数
successful_recommendations = 0
total_recommendations = 0

for user_id in users:
    user_interactions_subset = user_interactions[user_interactions['user_id'] == user_id]
    user_features = user_interactions_subset[['cat_id', 'seller_id', 'brand_id', 'age_range', 'gender']].values[0]
    tp_bandit.update(user_id, None, None)  # Initialize user parameters

    for _, interaction in user_interactions_subset.iterrows():
        item_id = [interaction['cat_id'], interaction['seller_id'], interaction['brand_id'], interaction['age_range'], interaction['gender']]
        reward = interaction['action_type']
        recommended_item = tp_bandit.recommend(user_id)
        tp_bandit.update(user_id, item_id, reward)

        if recommended_item is not None:
            total_recommendations += 1
            if (user_id, tuple(recommended_item)) in tp_bandit.interactions:
                successful_recommendations += 1

precision = successful_recommendations / total_recommendations
recall = successful_recommendations / len(tp_bandit.interactions)
f1_score = 2 * (precision * recall) / (precision + recall)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
