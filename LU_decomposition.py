try:
    import random
    import numpy
except ImportError:
    print("cant imprort module random")

AMOUNT_PERMUTATIONS = 0


def transpose_matrix(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]


def generating_Matrix(n, matrix):
    for i in range(n):
        for j in range(n):
            number_random = random.randint(0, 100)
            matrix[i][j] = number_random
    print("A = \n", end='')
    print(numpy.matrix(matrix), '\n')


def LU_decomosition(n, A, P):
    global AMOUNT_PERMUTATIONS
    for i in range(len(A)):
        pivot_value = 0
        pivot = -1
        for row in range(i, len(A)):
            if abs(A[row][i]) > pivot_value:
                pivot_value = A[row][i]
                pivot = row
        if pivot_value != 0:
            P[i], P[pivot] = P[pivot], P[i]
            A[i], A[pivot] = A[pivot], A[i]
            for j in range(i + 1, len(A)):
                A[j][i] /= A[i][i]
                for k in range(i + 1, len(A)):
                    A[j][k] -= A[j][i] * A[i][k]
    print("Матрица после LUP разложения", '\n')
    print(numpy.matrix(A))


# решаем уравнение Ax = b
def solving_equation(A, b):
    y = [0 for i in range(len(A))]
    for i in range(len(y)):
        y[i] = b[i] - sum([A[i][k] * y[k] for k in range(0, i)])
    x = [0 for i in range(len(A))]
    for i in range(len(x) - 1, -1, -1):
        x[i] = (y[i] - sum([A[i][k] * x[k] for k in range(i + 1, len(y))])) / A[i][i]
    return x


# находим обратную матрицу
def inverse_matrix(A, P):
    A_inverse = numpy.eye(len(A))
    E = numpy.eye(len(A))
    for i in range(len(A)):
        A_inverse[i] = solving_equation(A, E[P[i]])
    A_inverse = transpose_matrix(A_inverse)
    return A_inverse


def norma(matrix) -> float:
    max_sum = 0
    n = len(matrix)
    for i in range(n):
        temp_sum = 0
        for j in range(n):
            temp_sum += abs(matrix[j][i])
        if (temp_sum > max_sum):
            max_sum = temp_sum
    return temp_sum


# вычисляем число обусловленности по A_1
def conditional_number(A, A_inverse) -> float:
    number = norma(A) * norma(A_inverse)
    return number


# вычисляем определитель
def determinant(A):
    detA = 1
    for i in range(len(A)):
        detA *= A[i][i]
    detA = detA if AMOUNT_PERMUTATIONS % 2 == 0 else -detA
    return detA


def check_solution(A, x, b):
    print("Ax - b =", numpy.subtract(numpy.matmul(A, x), b))


def check_inversematrix(A, A_inverse):
    print("\n Левосторонее умножение: \n", numpy.matmul(A_inverse, A))
    print("\n правосторонее умножение: \n", numpy.matmul(A, A_inverse))


def main():
    matrix_size = int(input())
    A = [[0] * matrix_size for i in range(matrix_size)]
    generating_Matrix(matrix_size, A)
    A_copy = numpy.copy(A)
    P = [i for i in range(matrix_size)]

    LU_decomosition(matrix_size, A, P)

    b = [random.randint(0, 100) for i in range(len(P))]
    b_copy = b[:]
    for i in range(len(P)):
        b[i] = b_copy[P[i]]
    A_inv = inverse_matrix(A, P)
    x = solving_equation(A, b)
    print(numpy.matrix(b_copy))
    print("\n ОПРЕДЕЛИТЕЛЬ матрицы A:  \n ", determinant(A))
    print(" \n ****** Матрица b******: ")
    for i in range(len(b)):
        print("b_{} = ".format(i + 1), b[i])
    print("Решение матричного уравнения:  \n ")
    for i in range(len(x)):
        print("x_{} = ".format(i + 1), x[i])
    print(check_solution(A_copy, x, b_copy))
    print("\n Обратная матрица:  \n ", numpy.matrix(A_inv))
    check_inversematrix(A_copy, A_inv)
    print("\nЧисло обусловленности: ", conditional_number(A, A_inv))
    print("Матрица перестановок: \n", P)


main()
