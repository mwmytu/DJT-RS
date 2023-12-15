import pandas as pd
from sklearn.cluster import KMeans


user_interactions = pd.read_csv('Data/recommendation.csv')


features = ['cat_id', 'seller_id', 'brand_id', 'action_type', 'age_range', 'gender']


user_vectors = []
user_ids = []
for user_id, group in user_interactions.groupby('user_id'):
    user_ids.append(user_id)
    user_vector = group[features].mean().values
    user_vectors.append(user_vector)


kmeans = KMeans(n_clusters=10)  # 设置聚类的簇数为10
kmeans.fit(user_vectors)


user_labels = kmeans.labels_


cluster_user_matrix = pd.DataFrame(columns=['Cluster', 'User_Set'])


grouped_users = pd.Series(user_ids).groupby(user_labels)
for cluster, group in grouped_users:
    users = list(group)
    cluster_user_matrix = cluster_user_matrix.append({'Cluster': cluster, 'User_Set': users}, ignore_index=True)


print(cluster_user_matrix)

import numpy as np


user_parameters = np.array([1.0, 0.5, 0.2])
cluster_parameters = np.array([0.1, 0.3, 0.4])


data = [
    {'user_id': 1, 'cluster_id': 1, 'feedback': 1},
    {'user_id': 2, 'cluster_id': 2, 'feedback': 0},
    {'user_id': 3, 'cluster_id': 1, 'feedback': 1},

]


learning_rate = 0.01
epsilon = 1e-5
max_iterations = 1000



def loss_function(user_parameters, cluster_parameters, feedback):
    predicted_feedback = np.dot(user_parameters, cluster_parameters)
    return (predicted_feedback - feedback) ** 2



for iteration in range(max_iterations):

    user_gradient = np.zeros_like(user_parameters)
    cluster_gradient = np.zeros_like(cluster_parameters)

    total_loss = 0.0

    for sample in data:
        user_id = sample['user_id']
        cluster_id = sample['cluster_id']
        feedback = sample['feedback']

        predicted_feedback = np.dot(user_parameters[user_id], cluster_parameters[cluster_id])
        loss = loss_function(user_parameters[user_id], cluster_parameters[cluster_id], feedback)
        total_loss += loss

        user_gradient[user_id] += 2 * (predicted_feedback - feedback) * cluster_parameters[cluster_id]
        cluster_gradient[cluster_id] += 2 * (predicted_feedback - feedback) * user_parameters[user_id]


    user_parameters -= learning_rate * user_gradient
    cluster_parameters -= learning_rate * cluster_gradient


    print(f"Iteration {iteration + 1}: Loss = {total_loss}")


    if np.max(np.abs(user_gradient)) < epsilon and np.max(np.abs(cluster_gradient)) < epsilon:
        print("Converged!")
        break
