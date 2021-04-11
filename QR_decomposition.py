try:
    import random
    import numpy
    from math import sqrt
except ImportError:
    print("cant imprort module random")


# 1ый вариант QR разложения(метод вращений)
def QR_decomposition(A, size: int):
    Q = numpy.eye(size)
    R = A.copy()
    for i in range(size):
        for j in range(i + 1, size):
            s_sin = -R[j, i] / numpy.sqrt(R[i, i] ** 2 + R[j, i] ** 2)
            c_cos = R[i, i] / numpy.sqrt(R[i, i] ** 2 + R[j, i] ** 2)
            temp_Q = numpy.eye(size)
            temp_Q[i, i] = c_cos
            temp_Q[j, i] = s_sin
            temp_Q[i, j] = -s_sin
            temp_Q[j, j] = c_cos
            R = numpy.matmul(temp_Q, R)
            temp_Q[i, j] = s_sin
            temp_Q[j, i] = -s_sin
            Q = numpy.matmul(Q, temp_Q)
    return Q, R


def check_calculations(A, Q, R):
    print("Q = : \n", Q)
    print("R = : \n", R)
    print("\n A: \n ", numpy.matrix(A))
    print("\n Q * R = : \n", numpy.matmul(Q, R))
    A_afterQR = numpy.matmul(Q, numpy.transpose(Q))
    print("\n Q * Q^T = : \n", A_afterQR)
    print("Решение Ax = b:", numpy.linalg.solve(A_afterQR, b=[random.randint(0, 100) for i in range(len(A))]))


def main():
    matrix_size = int(input())
    A = numpy.matrix([
        [random.randint(1, 100000) for j in range(matrix_size)] for i in range(matrix_size)])
    Q, R = QR_decomposition(A, matrix_size)
    check_calculations(A, Q, R)


main()
