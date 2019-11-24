import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fmin_l_bfgs_b

from generators import ContinuousRandomNumberGenerator
from lab1_2 import get_end_point_for_y


A = 0
B = 1
STEP = 0.05
N = 10_000 
BIN = 10

def is_seccess_constraint(value):
    return value >= 0 and value <= 1

def f_xy(x, y):
    return 12/7*(x**2 + y/2)

def f_x(x):
    return (12*x**2 + 3)/7 
    
def f_x_inv(x):
    return (7*x - 3)/12

def f_y(y):
    return (4 + 6*y)/7

def f_y_inv(y):
    return (7*y - 4)/6

class Generator2D:
    def __init__(self, first_genenerator:ContinuousRandomNumberGenerator, second_generator, a, b):
        self.__first = first_genenerator
        self.__second = second_generator
        self.__a = a
        self.__b = b
        print(self.__b, self.__first.func(self.__b), self.__first.func(self.__a))

    def next(self):
        return (
                self.__first.next(self.__b, 
                                  self.__first.func(self.__b), 
                                  a_y=self.__first.func(self.__a)), 
                self.__second.next(self.__b, 
                                   self.__second.func(self.__b), 
                                   a_y=self.__second.func(self.__a))
               )

    def get_iterator(self, count):
        for _ in range(count):
            yield self.next()


if __name__ == '__main__':

    end_point_for_f_x = get_end_point_for_y(f_x, A, B)
    end_point_for_f_y = get_end_point_for_y(f_y, A, B)
    
    first_sub_generator = ContinuousRandomNumberGenerator(f_x, end_point_for_f_x)
    second_sub_generator = ContinuousRandomNumberGenerator(f_y, end_point_for_f_y)

    generator = Generator2D(first_sub_generator, second_sub_generator, A, B)

    values = list(generator.get_iterator(N))

    all_x_for_x = np.arange(A, B, STEP)
    all_y_for_x = [f_x(x) for x in all_x_for_x]

    all_x_for_y = np.arange(A, B, STEP)
    all_y_for_y = [f_y(y) for y in all_x_for_y]

    actual_values_x = [value[0] for value in values]
    actual_values_y = [value[1] for value in values]

    plt.plot(all_x_for_x, all_y_for_x)
    plt.hist(actual_values_x,
             weights=np.zeros_like(actual_values_x) + 1. / N,
             bins=BIN)
    plt.show()

    plt.plot(all_x_for_y, all_y_for_y)
    plt.hist(actual_values_y,
             weights=np.zeros_like(actual_values_y) + 1. / N,
             bins=BIN)
    plt.show()

