import math
import numpy as np
import time
import random


def F(x):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = x.tolist()
    return np.transpose(np.mat([
        math.cos(x2 * x1) - math.exp(-3 * x3) + x4 * x5 ** 2 - x6 - math.sinh(
            2 * x8) * x9 + 2 * x10 + 2.000433974165385440,
        math.sin(x2 * x1) + x3 * x9 * x7 - math.exp(-x10 + x6) + 3 * x5 ** 2 - x6 * (x8 + 1) + 10.886272036407019994,
        x1 - x2 + x3 - x4 + x5 - x6 + x7 - x8 + x9 - x10 - 3.1361904761904761904,
        2 * math.cos(-x9 + x4) + x5 / (x3 + x1) - math.sin(x2 ** 2) + math.cos(
            x7 * x10) ** 2 - x8 - 0.1707472705022304757,
        math.sin(x5) + 2 * x8 * (x3 + x1) - math.exp(-x7 * (-x10 + x6)) + 2 * math.cos(x2) - 1.0 / (
                -x9 + x4) - 0.3685896273101277862,
        math.exp(x1 - x4 - x9) + x5 ** 2 / x8 + math.cos(3 * x10 * x2) / 2 - x6 * x3 + 2.0491086016771875115,
        x2 ** 3 * x7 - math.sin(x10 / x5 + x8) + (x1 - x6) * math.cos(x4) + x3 - 0.7380430076202798014,
        x5 * (x1 - 2 * x6) ** 2 - 2 * math.sin(-x9 + x3) + 0.15e1 * x4 - math.exp(
            x2 * x7 + x10) + 3.5668321989693809040,
        7 / x6 + math.exp(x5 + x4) - 2 * x2 * x8 * x10 * x7 + 3 * x9 - 3 * x1 - 8.4394734508383257499,
        x10 * x1 + x9 * x2 - x8 * x3 + math.sin(x4 + x5 + x6) * x7 - 0.78238095238095238096]))


def J(x):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = x.tolist()
    return np.asarray([[-x2 * math.sin(x2 * x1), -x1 * math.sin(x2 * x1), 3 * math.exp(-3 * x3), x5 ** 2, 2 * x4 * x5,
                        -1, 0, -2 * math.cosh(2 * x8) * x9, -math.sinh(2 * x8), 2],
                       [x2 * math.cos(x2 * x1), x1 * math.cos(x2 * x1), x9 * x7, 0, 6 * x5,
                        -math.exp(-x10 + x6) - x8 - 1, x3 * x9, -x6, x3 * x7, math.exp(-x10 + x6)],
                       [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                       [-x5 / (x3 + x1) ** 2, -2 * x2 * math.cos(x2 ** 2), -x5 / (x3 + x1) ** 2,
                        -2 * math.sin(-x9 + x4),
                        1.0 / (x3 + x1), 0, -2 * math.cos(x7 * x10) * x10 * math.sin(x7 * x10), -1,
                        2 * math.sin(-x9 + x4), -2 * math.cos(x7 * x10) * x7 * math.sin(x7 * x10)],
                       [2 * x8, -2 * math.sin(x2), 2 * x8, 1.0 / (-x9 + x4) ** 2, math.cos(x5),
                        x7 * math.exp(-x7 * (-x10 + x6)), -(x10 - x6) * math.exp(-x7 * (-x10 + x6)), 2 * x3 + 2 * x1,
                        -1.0 / (-x9 + x4) ** 2, -x7 * math.exp(-x7 * (-x10 + x6))],
                       [math.exp(x1 - x4 - x9), -1.5 * x10 * math.sin(3 * x10 * x2), -x6, -math.exp(x1 - x4 - x9),
                        2 * x5 / x8, -x3, 0, -x5 ** 2 / x8 ** 2, -math.exp(x1 - x4 - x9),
                        -1.5 * x2 * math.sin(3 * x10 * x2)],
                       [math.cos(x4), 3 * x2 ** 2 * x7, 1, -(x1 - x6) * math.sin(x4),
                        x10 / x5 ** 2 * math.cos(x10 / x5 + x8),
                        -math.cos(x4), x2 ** 3, -math.cos(x10 / x5 + x8), 0, -1.0 / x5 * math.cos(x10 / x5 + x8)],
                       [2 * x5 * (x1 - 2 * x6), -x7 * math.exp(x2 * x7 + x10), -2 * math.cos(-x9 + x3), 1.5,
                        (x1 - 2 * x6) ** 2, -4 * x5 * (x1 - 2 * x6), -x2 * math.exp(x2 * x7 + x10), 0,
                        2 * math.cos(-x9 + x3),
                        -math.exp(x2 * x7 + x10)],
                       [-3, -2 * x8 * x10 * x7, 0, math.exp(x5 + x4), math.exp(x5 + x4),
                        -7.0 / x6 ** 2, -2 * x2 * x8 * x10, -2 * x2 * x10 * x7, 3, -2 * x2 * x8 * x7],
                       [x10, x9, -x8, math.cos(x4 + x5 + x6) * x7, math.cos(x4 + x5 + x6) * x7,
                        math.cos(x4 + x5 + x6) * x7, math.sin(x4 + x5 + x6), -x3, x2, x1]])


def find_max_element(A):
    element = [0, 0]
    for i in range(len(A)):
        for j in range(len(A)):
            if (A[i, j] >= A[element[0], element[1]]):
                element = [i, j]
    return element


def filling_P(P_copy):
    size = len(P_copy)
    E = np.eye(size)
    P = np.zeros((size, size))
    for i in range(size):
        P[i] = E[P_copy[i]]
    return P


def filling_Q(Q_copy):
    size = len(Q_copy)
    E = np.eye(size)
    Q = np.zeros((size, size))
    for i in range(size):
        Q[i] = E[Q_copy[i]]
    return np.transpose(Q)


def LUPQ_decompose(A, size):
    global counter
    P_copy = [i for i in range(size)]
    Q_copy = [i for i in range(size)]
    for i in range(size):
        element = [k + i for k in find_max_element(A[i:, i:])]
        A[[i, element[0]]] = A[[element[0], i]]
        P_copy[i], P_copy[element[0]] = P_copy[element[0]], P_copy[i]

        A[:, [i, element[1]]] = A[:, [element[1], i]]
        Q_copy[i], Q_copy[element[1]] = Q_copy[element[1]], Q_copy[i]
        for j in range(i + 1, A.shape[0]):
            A[j, i] = A[j, i] / A[i, i]
            counter += 1
            for k in range(i + 1, A.shape[1]):
                counter += 1
                A[j, k] -= A[j, i] * A[i, k]

    L = np.copy(A)
    U = np.copy(A)

    for i in range(size):
        L[i, i] = 1
        L[i, i + 1:] = 0
    for i in range(1, size):
        U[i, :i] = 0

    P_copy = filling_P(P_copy)
    Q_copy = filling_Q(Q_copy)

    return L, U, P_copy, Q_copy


def solve_equation(L, U, P, Q, b):
    global counter
    b = np.dot(P, b)
    y = np.matrix(np.zeros([L.shape[0], 1]))
    for i in range(y.shape[0]):
        counter += L.shape[0]
        y[i, 0] = b[i, 0] - L[i, :i] * y[:i]
    x = np.matrix(np.zeros([L.shape[0], 1]))
    for i in range(1, x.shape[0] + 1):
        counter += U.shape[0]
        x[-i, 0] = (y[-i] - U[-i, -i:] * x[-i:, 0]) / U[-i, -i]
    return np.dot(Q, np.squeeze(np.asarray(x)))


def Newton_method(epsilon, k, is_hybrid):
    amount_operations = 0
    print("НАчальное приближение \n")
    x = np.array([0.5, 0.5, 1.5, -1.0, -0.2, 1.5, 0.5, -0.5, 1.5, -1.5])
    print(x)
    jacobian = J(x)
    size = 10
    while (amount_operations < 10000):
        if (is_hybrid and amount_operations % k == 0 or not is_hybrid and amount_operations < k):
            jacobian = J(x)
            L, U, P, Q = LUPQ_decompose(jacobian, size)
        print("x" + str(amount_operations), " = ", x)
        xk = np.add(x, solve_equation(L, U, P, Q, -F(x)))
        delta = np.linalg.norm(np.subtract(x, xk))
        print("delta", delta)
        amount_operations += 1
        x = np.asarray(xk)
        if delta < epsilon:
            return x


counter = 0
x = Newton_method(2e-16, 2,True)
start = time.time()
print("\n **** Время затраченное: ", time.time() * - start, "Количество операций ", counter, "\n \n")
print("F(x) = ", F(x))
