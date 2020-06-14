import pdb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D


def extract_data():
    with open('./input03.txt', 'r') as F:
        lines = list(map(int, F.readline().split()))
        X_train = []
        Y_train = []
        for _ in range(lines[1]):
            train_data = list(map(float, F.readline().strip().split()))
            X_train.append(train_data[:2])
            Y_train.append(train_data[2])

        X_test = []
        test_lines = int(F.readline())
        for _ in range(test_lines):
            test_data = list(map(float, F.readline().strip().split()))
            X_test.append(test_data)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)

    Y_test = []
    with open('./output03.txt', 'r') as F:
        for _ in range(test_lines):
            test_val = float(F.readline().strip().split()[0])
            Y_test.append(test_val)
    return X_train, Y_train, X_test, Y_test


def LinearRegression_numpy(X_train, X_test, Y_train):
    x_train = X_train.copy()
    x_test = X_test.copy()
    y_train = Y_train.copy()
    x_train = np.column_stack((np.ones([x_train.shape[0], 1]), x_train))
    x_test = np.column_stack((np.ones([x_test.shape[0], 1]), x_test))

    coef = np.matmul(np.matmul(np.linalg.pinv(
        np.matmul(x_train.T, x_train)), x_train.T), y_train)

    Y_pred = np.matmul(coef, x_test.T)
    return Y_pred


def LinearRegression_sklearn(X_train, Y_train, X_test):
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, Y_train)
    Yhattest = lin_reg.predict(X_test)
    return Yhattest


X_train, Y_train, X_test, Y_test = extract_data()
print(LinearRegression_numpy(X_train, X_test, Y_train))
print(LinearRegression_sklearn(X_train, Y_train, X_test))

Yhat = LinearRegression_sklearn(X_train, Y_train, X_train)
# Visualize and compare output
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[:, 0], X_train[:, 1], Y_train, c='b', marker='x',
           label='X_train vs Y_train', s=10)
ax.scatter(X_train[:, 0], X_train[:, 1], Yhat, c='r',
           marker='s', label='X_train vs Yhat', s=10)
plt.legend(loc='lower right')
plt.title("Without polynomial features")
plt.show()

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X_train)
X_poly_test = poly.fit_transform(X_test)
print(LinearRegression_numpy(X_poly, X_poly_test, Y_train))
print(LinearRegression_sklearn(X_poly, Y_train, X_poly_test))

Yhat = LinearRegression_sklearn(X_poly, Y_train, X_poly)
# Visualize and compare output
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[:, 0], X_train[:, 1], Y_train, c='b', marker='x',
           label='X_train vs Y_train', s=10)
ax.scatter(X_train[:, 0], X_train[:, 1], Yhat, c='r',
           marker='s', label='X_train vs Yhat', s=10)
plt.legend(loc='lower right')
plt.title("With polynomial features")
plt.show()
