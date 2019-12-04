import numpy as np
import matplotlib.pyplot as plt
import seaborn

from scipy.optimize import fmin_l_bfgs_b
from math import sqrt

from generators import ContinuousRandomNumberGenerator, BaseRandomGeneretor
from lab1_2 import get_end_point_for_y
from scipy.stats import norm, chi2


A = 0
B = 1
STEP = 0.05
N = 10_000
BIN = 10

M1_THEORY = 9/14
D1_THEORY = 1/7*(12/5 + 1) - M1_THEORY**2



M2_THEORY = 4/7
D2_THEORY = 3/14 + 4/21 - M2_THEORY**2

COR = (1 - M1_THEORY*M2_THEORY)/(sqrt(D1_THEORY)*sqrt(D2_THEORY))

def f_xy(x, y):
    return 12/7*(x**2 + y/2)

def f_x(x):
    return (12*x**2 + 3)/7 

def f_y_x(x, y):
    return f_xy(x, y)/f_x(x)
    
def f_y(y):
    return (4 + 6*y)/7

def f_x_y(y, x):
    return f_xy(x, y)/f_y(y) 

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


def analyze(f1, f2, A, B, n):
    generator = Generator2D(f1, f2, A, B)

    values = tuple(zip(*generator.get_iterator(n)))
    x_range = np.arange(A, B, 0.05)
    y_range = [f1(x) for x in x_range]

    plt.subplot(1, 2, 1)
    plt.plot(x_range, y_range)
    plt.subplot(1, 2, 2)
    plt.hist(values[0], 20, weights=np.zeros_like(values[0]) + 1./N)
    plt.show()
    
    m1 = sum(values[0])/N
    d1 = sum((value - m1)**2 for value in values[0])/(N + 1)
    
    m2 = sum(values[1])/N
    d2 = sum((value - m2)**2 for value in values[1])/(N + 1)

    cov = sum(val[0]*val[1] - m1*m2 for val in values)
    cor = sum((val[0]-m1)*(val[1]-m2) for val in values) / sqrt(d1*d2)


    return ((m1, m2), (d1, d2), cov, cor)
    

def get_interval_assessment_M(m, d, n):
    s = sqrt(d)
    k = s/sqrt(N-1)*norm.ppf(0.99)

    return m-k, m+k

def get_interval_assessment_D(m, d, n):
    return (
            n * d/chi2.isf((1-0.99)/2, n-1),
            n * d/chi2.isf((1+0.99)/2, n-1)
            )



if __name__ == '__main__':
    m1, d1, cov_1, cor_1 = analyze(f_x, f_y_x, A, B, N)
    c1, c2 = get_interval_assessment_M(m1[0], d1[0], N)
    z1, z2 = get_interval_assessment_D(m1[0], d1[0], N)

    print('M)', M1_THEORY, m1[0])
    print('D)', D1_THEORY, d1[0])
    print(f'{c1} < {m1[0]} < {c2}')
    print(f'{z1} < {d1[0]} < {z2}')
    
    m2, d2, cov_2, cor_2 = analyze(f_y, f_x_y, A, B, N)
    c1, c2 = get_interval_assessment_M(m2[0], d2[0], N)
    z1, z2 = get_interval_assessment_D(m2[0], d2[0], N)

    print('M)', M2_THEORY, m2[0])
    print('D)', D2_THEORY, d2[0])
    print(f'{c1} < {m2[0]} < {c2}')
    print(f'{z1} < {d2[0]} < {z2}')

    print(cor_1, '=', cor_2)
    print(COR)
    print(cov_1, '=', cov_2)