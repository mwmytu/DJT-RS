import numpy as np


num_users = 100
num_items = 50
latent_dim = 5


user_item_matrix = np.random.rand(num_users, num_items)


def matrix_factorization_SGD(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    # 计算预测评分的误差
                    eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                    # 使用梯度下降更新 P 和 Q
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

        eR = np.dot(P, Q)
        e = 0

        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)
                    for k in range(K):

                        e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))

        if e < 0.001:
            break
    return P, Q.T


P = np.random.rand(num_users, latent_dim)
Q = np.random.rand(num_items, latent_dim)


P, Q = matrix_factorization_SGD(user_item_matrix, P, Q, latent_dim)


predicted_user_item_matrix = np.dot(P, Q.T)
