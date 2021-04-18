try:
    import numpy as np
    import random as rand
except ImportError:
    print("cant imprort module random")

RANK = 0


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
    global RANK
    RANK = size

    P_copy = [i for i in range(size)]
    Q_copy = [i for i in range(size)]
    for i in range(size):
        element = [k + i for k in find_max_element(A[i:, i:])]
        if (A[element[0], element[1]] < 1e-13):
            RANK = i
            break
        A[[i, element[0]]] = A[[element[0], i]]
        P_copy[i], P_copy[element[0]] = P_copy[element[0]], P_copy[i]

        A[:, [i, element[1]]] = A[:, [element[1], i]]
        Q_copy[i], Q_copy[element[1]] = Q_copy[element[1]], Q_copy[i]
        for j in range(i + 1, A.shape[0]):
            A[j, i] = A[j, i] / A[i, i]
            for k in range(i + 1, A.shape[1]):
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


def solve_equation(L, U, b):
    y = np.matrix(np.zeros([L.shape[0], 1]))
    for i in range(y.shape[0]):
        y[i, 0] = b[i, 0] - L[i, :i] * y[:i]
    x = np.matrix(np.zeros([L.shape[0], 1]))
    for i in range(1, x.shape[0] + 1):
        if (U[-i, -i] < 1e-13):
            if (y[-i] > 1e-13):
                raise BaseException("No solution")
            else:
                continue
        x[-i, 0] = (y[-i] - U[-i, -i:] * x[-i:, 0]) / U[-i, -i]
    return x


def check_solution(A, x, b):
    print("Ax - b = \n", np.subtract(np.matmul(A, x), b))


def check_matrices(L, U, P, Q, A):
    print("Матрица L \n", np.matrix(L))
    print("Матрица U \n", np.matrix(U))
    print("Матрица LU \n", np.matrix(np.matmul(L, U)))
    print("Матрица A: \n", np.matrix(A))
    print("Матрица PAQ \n", np.matrix(np.matmul(P, np.matmul(A, Q))))


def main():
    size = rand.randint(4, 7)
    # size = 5
    A = np.matrix([[rand.randint(0, 10) + 0.0 for j in range(size)]
                   for i in range(size)])
    A[:, 1] = A[:, 0] * 3 + A[:, 3] * 2
    A[:, 2] = A[:, 3] * 3 + A[:, 0] * 2

    b = np.matrix([rand.randint(0, 10) for i in range(size)])
    b = np.transpose(b)
    b = np.matmul(A, b)

    P = [i for i in range(size)]
    Q = [i for i in range(size)]

    L = np.eye(size)
    U = np.zeros(size)

    L = L.astype(np.float64)
    U = U.astype(np.float64)
    A = A.astype(np.float64)
    A_copy = np.copy(A)

    L, U, P, Q = LUPQ_decompose(A, size)

    check_matrices(L, U, P, Q, A_copy)
    print("РАЗМЕР  = ", f"{size} x {size}")
    print("РАНГ  = ", RANK)
    try:
        x = np.matmul(Q, solve_equation(L, U, np.matmul(P, b)))
        print("b = \n", np.matrix(b))
        print("x = \n", (np.matrix(x)))
        check_solution(A_copy, x, b)
    except BaseException as e:
        print(e)


main()
