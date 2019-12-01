import numpy as np
import matplotlib.pyplot as plt
import seaborn

from scipy.optimize import fmin_l_bfgs_b

from generators import ContinuousRandomNumberGenerator, BaseRandomGeneretor
from lab1_2 import get_end_point_for_y
from collections import defaultdict
from pandas import DataFrame

A = 0
B = 1
STEP = 0.05
N = 100_000
BIN = 10

M1_THEORY = 9/14
D1_THEORY = 5/7 - (9/14)**2

def is_seccess_constraint(value):
    return value >= 0 and value <= 1

def f_xy(x, y):
    return 12/7*(x**2 + y/2)

def f_x(x):
    return (12*x**2 + 3)/7 

def f_y_x(x, y):
    return f_xy(x, y) / f_x(x)
    
def f_x_inv(x):
    return ((7*x - 3)/12)**0.5

def f_y(y):
    return (4 + 6*y)/7

def f_y_inv(y):
    return (7*y - 4)/6




class Generator2D:

    def __init__(self, f1, f2, a1=None, b1=None, a2=None, b2=None):
        self.__generator = BaseRandomGeneretor()
        self.__f1 = f1
        self.__f2 = f2
        self.__a1 = a1 if a1 is not None else 0
        self.__b1 = b1 if b1 is not None else 1
        self.__a2 = a2 if a2 is not None else self.__a1
        self.__b2 = b2 if b2 is not None else self.__b1
        self.__f1_max = f1(self.__b1)

    def next(self):
        x = self.__generate_first_value()
        y = self.__generate_second_value(x)
        
        return x, y

    def __generate_first_value(self):
        while True:
            x = self.__a1 + (self.__b1 - self.__a1)*self.__generator.next()
            y = self.__f1_max * self.__generator.next()
            if self.__f1(x) >= y:
                return x

    def __generate_second_value(self, first_value):
        while True:
            x = self.__a2 + (self.__b2 - self.__a1)*self.__generator.next()
            y_max = fmin_l_bfgs_b(lambda y: self.__f2(first_value, y),
                                  (self.__b2 - self.__a2) / 2,
                                  bounds=[(self.__a2, self.__b2)],
                                  approx_grad=True)[0][0]
            
            y = y_max * self.__generator.next()
            if self.__f2(first_value, x) >= y:
                return x 
        

    def get_iterator(self, count: int):
        for _ in range(count):
            yield self.next()


        
if __name__ == '__main__':
    generator = Generator2D(f_x, f_y_x, A, B)
    
    values = list(zip(*generator.get_iterator(N)))

    x_range = np.arange(A, B, 0.05)
    y_range = [f_x(x) for x in x_range]

    plt.subplot(1, 2, 1)
    plt.plot(x_range, y_range)
    plt.subplot(1, 2, 2)
    plt.hist(values[0], 20, weights=np.zeros_like(values[0]) + 1./N)
    plt.show()
    
    m1 = sum(values[0]) / N
    d1 = sum(value**2 - (M1_THEORY**2) for value in values[0])
    print(M1_THEORY, m1)
    print(D1_THEORY, d1)
