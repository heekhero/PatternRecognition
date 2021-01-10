import numpy as np
import copy

from tqdm import tqdm
from matplotlib import pyplot as plt


def generate_data(k=4, size=1000, m=2):

    _size = size // k

    data = None

    means = np.array([[1,1],
                      [1,-1],
                      [-1,1],
                      [-1,-1]]) + np.random.rand(k, m) * 0.01

    for i in range(k):
        mean = means[i]
        covariance = np.identity(2) * (0.1 + np.random.rand() * 0.1)

        _data = np.random.multivariate_normal(mean, covariance, _size)
        if data is None:
            data = _data
        else:
            data = np.vstack((data, _data))

    return data


def softmax(x):
    e_x = np.exp(x)
    prob_x = e_x / np.sum(e_x)
    assert np.sum(prob_x) == 1
    return prob_x

m = 2
k = 4
size = 1000

data = generate_data(k, size, m)

alpha = softmax(np.random.randn(k))
mu = np.random.randn(k, m)
sigma = np.zeros(shape=(k, m, m))
for _k in range(k):
    sigma[_k] = np.identity(2) * (0.1 + np.random.rand() * 0.1)

def density(xi, mu, sigma):

    d = mu.shape[0]

    _sigma = np.linalg.inv(sigma)

    coefficient = 1.0 / ((2 * np.pi) ** (d/2) * (np.linalg.det(sigma) ** 0.5))
    body = np.e ** (-0.5 * np.matmul(np.matmul((xi-mu).reshape(1, -1), _sigma), (xi-mu).reshape(-1, 1)))

    return coefficient * body

def mse(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

plt.scatter(mu[:, 0], mu[:, 1], c='red')

for t in range(100):
    old_alpha = copy.deepcopy(alpha)
    old_mu = copy.deepcopy(mu)
    old_sigma = copy.deepcopy(sigma)

    #calculate Q
    q = np.zeros(shape=(size, k))
    for i in range(size):
        sum_k = 0.0
        for _k in range(k):
            denst = density(data[i], mu[_k], sigma[_k])
            q[i, _k] = alpha[_k] * denst
            sum_k += q[i, _k]

        q[i] /= sum_k

    #update mu
    to_data = q.reshape(q.shape[0], q.shape[1], -1) * data.reshape(data.shape[0], -1, data.shape[1])
    mu = np.sum(to_data, axis=0) / np.sum(q, axis=0).reshape(q.shape[1], -1)
    print()

    #updaye sigma
    sum_sigma = np.zeros(shape=(size, k, m, m))
    for i in range(size):
        for _k in range(k):
            sum_sigma[i, _k] = np.matmul((data[i] - mu[_k]).reshape(-1, 1), (data[i] - mu[_k]).reshape(1, -1)) * q[i][_k]

    sigma = np.sum(sum_sigma, axis=0) / np.sum(q, axis=0).reshape(k, 1, 1)

    #update alpha
    alpha = np.sum(q, axis=0) / size

    loss = mse(alpha, old_alpha) + mse(mu, old_mu) + mse(sigma, old_sigma)

    print('iter : {}  loss : {}'.format(t, loss))


plt.scatter(data[:, 0], data[:, 1])
plt.scatter(mu[:, 0], mu[:, 1], c='green')
plt.axis()
plt.title("scatter")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

q = np.zeros(shape=(size, k))
for i in range(size):
    sum_k = 0.0
    for _k in range(k):
        denst = density(data[i], mu[_k], sigma[_k])
        q[i, _k] = alpha[_k] * denst
        sum_k += q[i, _k]

    q[i] /= sum_k

prediction = np.argmax(q, axis=1)

plt.figure()
plt.scatter(data[:, 0], data[:, 1], c=prediction)
plt.axis()
plt.title("clustering")
plt.xlabel("x")
plt.ylabel("y")
plt.show()






