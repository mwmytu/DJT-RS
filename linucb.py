import numpy as np
import pandas as pd


class LinUCB:

	def __init__(self, alpha=0.25, r1=1, r0=0, d=6):
		self.alpha = alpha
		self.r1 = r1
		self.r0 = r0
		self.d = d
		self.Aa = {}
		self.AaI = {}
		self.ba = {}
		self.a_max = 0
		self.theta = {}
		self.x = None
		self.xT = None

	def set_articles(self, articles):
		for article in articles:
			self.Aa[article] = np.identity(self.d)
			self.ba[article] = np.zeros((self.d, 1))
			self.AaI[article] = np.identity(self.d)
			self.theta[article] = np.zeros((self.d, 1))

	def update(self, reward):
		if reward == -1:
			pass
		elif reward == 1 or reward == 0:
			if reward == 1:
				r = self.r1
			else:
				r = self.r0
			self.Aa[self.a_max] += np.dot(self.x, self.xT)
			self.ba[self.a_max] += r * self.x
			self.AaI[self.a_max] = np.linalg.solve(self.Aa[self.a_max], np.identity(self.d))
			self.theta[self.a_max] = np.dot(self.AaI[self.a_max], self.ba[self.a_max])
		else:
			pass

	def recommend(self, timestamp, user_features, articles):
		xaT = np.array([user_features])
		xa = np.transpose(xaT)
		AaI_tmp = np.array([self.AaI[article] for article in articles])
		theta_tmp = np.array([self.theta[article] for article in articles])
		self.a_max = articles[
			np.argmax(np.dot(xaT, theta_tmp) + self.alpha * np.sqrt(np.dot(np.dot(xaT, AaI_tmp), xa)))]
		self.x = xa
		self.xT = xaT
		return self.a_max



item_features = pd.read_csv('DJT-RS/Data/ITEM.csv')


user_interactions = pd.read_csv('DJT-RS/Data/ITEM.csv')


items = item_features['item_id'].unique()
user_features = [1, 0, 1, 0, 1, 0]


linucb = LinUCB(alpha=0.25, r1=1, r0=0, d=6)
linucb.set_articles(items)


successful_recommendations = 0
total_recommendations = 0


interactions = {}

for _, interaction in user_interactions.iterrows():
	user_id = interaction['user_id']
	item_id = interaction['item_id']
	reward = interaction['action_type']
	recommended_item = linucb.recommend(timestamp=None, user_features=user_features, articles=items)
	linucb.update(reward)
	print(f"User {user_id} recommended item: {recommended_item}")

	# 计算准确率、召回率和F1得分
	if user_id not in interactions:
		interactions[user_id] = set()
	interactions[user_id].add(item_id)
	if recommended_item == item_id:
		successful_recommendations += 1
	total_recommendations += 1


precision = successful_recommendations / total_recommendations
recall = successful_recommendations / len(interactions)
f1_score = 2 * (precision * recall) / (precision + recall)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
