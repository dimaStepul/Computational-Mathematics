import math

def main():
    arguments_list = [0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30]
    temp = 0.
    print("arguments:                      exact:                                          approx:                                           error")
    for i in range(11):
        temp = arguments_list[i]
        print_list(temp, f_exact(temp), f_approx(temp))



#реализуем вычисление корня при помощи формулы герона
def calculation_sqrt(argument):
    x0 = float(max(argument, 1)) #находим первый член в рекуррентной формули
    epsilon = 0.000001 / 4.41  #задаем точность
    x_i = x0
    x_j = (x_i + (argument / x_i)) / 2
    difference = math.fabs(x_j - x_i)
    x_i = x_j
    while (difference >= epsilon):
        x_j = (x_i + (argument / x_i)) / 2
        difference = math.fabs(x_j - x_i)
        x_i = x_j
    return x_j
#рекурретно вычисляем синус
def calculation_sin(argument):
    if (argument < 0 ):
        argument *= -1
        return (-1)*calculation_sin(argument)
    argument = argument % (2 * math.pi)
    # ниже я привожу аргумент к нужному отрезку(от -pi/4 до pi/4)
    if(argument > math.pi / 4 and argument < 3 * math.pi / 4 ):
        argument = math.fabs((math.pi / 2) - argument)
        return calculation_cos(argument)
    elif (argument > 3 * math.pi / 4 and argument < 5 * math.pi / 4):
        if (argument > math.pi):
            argument = math.fabs(math.pi - argument)
            return (-1)*calculation_sin(argument)
        else:
            argument = math.fabs(math.pi - argument)
            return calculation_sin(argument)
    elif (argument > 5 * math.pi / 4 and argument < 7 * math.pi / 4):
        argument = math.fabs((3  * math.pi / 2) - argument)
        return (-1)*calculation_cos(argument)
    elif (math.fabs(argument) <= math.pi / 4):
        result_of_sum = 0.
        t = argument
        result_of_sum += argument #прибавили к сумме нулевой член ряда(k == 0)
        value_u = argument;
        epsilon = 0.000001 / 3.15
        k = 2
        t = argument * argument
        while (math.fabs(value_u) >= epsilon):
            value_u *= ((-t) / (k * (k + 1)));
            result_of_sum += value_u;
            k += 2
        return result_of_sum;

#рекурретно вычисляем косинус
def calculation_cos(argument):
    if (argument < 0):
        argument *= -1
    argument = argument % (2 * math.pi)
    # ниже я привожу аргумент к нужному отрезку(от -pi/4 до pi/4)
    if (argument > math.pi / 4 and argument < 3 * math.pi / 4):
        if(argument > math.pi / 2):
            argument = math.fabs(argument - math.pi/2)
            return (-1)*calculation_sin(argument)
        else:
            argument = math.fabs(argument - math.pi / 2)
            return calculation_sin(argument)
    elif (argument > 3 * math.pi / 4 and argument < 5 * math.pi / 4):
        argument = math.fabs(math.pi - argument)
        return (-1)*calculation_cos(argument)
    elif (argument > 5 * math.pi / 4 and argument < 7 * math.pi / 4):
        if(argument < 3 * math.pi / 2):
            argument = math.fabs(3 *math.pi / 2 - argument)
            return (-1) * calculation_sin(argument)
        else:
            argument = math.fabs(3 * math.pi / 2 - argument)
            return calculation_sin(argument)
    elif (math.fabs(argument) <= math.pi / 4):
        t = argument
        epsilon = 0.000001 / 3.15
        k = 2
        value_u = 1;
        result_of_sum = 1. #прибавили к сумме нулевой член ряда(k == 0)
        while (math.fabs(value_u)>= epsilon):
            value_u *= -(t * t)/(k * (k - 1)) ;
            result_of_sum += value_u;
            k += 2
        return result_of_sum;

def f_exact(x):
    return  (math.sqrt(1 + x * x) * (math.sin(3 * x + 0.1) + math.cos(2 * x + 0.3)))


def f_approx(x):
    return (calculation_sqrt(x * x + 1.0) * (calculation_sin(x * 3. + 0.1 ) + calculation_cos(x * 2. + 0.3)))

def print_list(argument, exact, approx):
    error = math.fabs(exact - approx)
    print(argument, "                       ", exact, "                              ", approx,
          "                        ", error)


# Вызов функции main
main()
