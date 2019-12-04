from math import sqrt
from generators import BaseRandomGeneretor

import matplotlib.pyplot as plt
import numpy as np
import itertools


DISTRIBUTION = [
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 3, 2, 3, 3, 3, 4, 4, 4, 1, 1],
          [1, 4, 5, 5, 5, 5, 6, 5, 5, 1, 1],
          [1, 4, 5, 5, 5, 8, 9, 9, 9, 5, 1],
          [1, 4, 5, 6, 7, 8, 9, 9, 9, 8, 7],
          [1, 4, 5, 6, 7, 8, 9, 9, 9, 6, 1],
          [1, 1, 4, 4, 4, 4, 4, 8, 8, 8, 1],
          [1, 1, 2, 2, 2, 2, 3, 5, 5, 5, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
          ]


ALL_SUM = sum(sum(line) for line in DISTRIBUTION)

N = 10_000

M1_THEORY = 0
M2_THEORY = 0
M3_THEORY = 0
D1_THEORY = 0
D2_THEORY = 0



for i, line in enumerate(DISTRIBUTION):
    for j, val in enumerate(line):
        DISTRIBUTION[i][j]/=ALL_SUM

for i in range(len(DISTRIBUTION)):
    for j in range(len(DISTRIBUTION[i])):
        M1_THEORY += DISTRIBUTION[i][j] * j
        M2_THEORY += DISTRIBUTION[i][j] * i
        M3_THEORY += DISTRIBUTION[i][j] * i * j

       

for i in range(len(DISTRIBUTION)):
    for j in range(len(DISTRIBUTION[i])):
        D1_THEORY += DISTRIBUTION[i][j] * (j - M1_THEORY)**2
        D2_THEORY += DISTRIBUTION[i][j] * (i - M2_THEORY)**2


COR = (M3_THEORY - M1_THEORY*M2_THEORY)/sqrt(D1_THEORY*D2_THEORY)


class Generato2D:

    def __init__(self, distribution):
        self.__generator = BaseRandomGeneretor()
        self.__dist = distribution

    def __get_p(self, i=None, j=None):
        if i is None:
            return sum(self.__dist[j])
        if j is None:
            return sum(line[i] for line in self.__dist)
        return self.__dist[j][i]


    def next(self):
        x, y = self.__generator.next(), self.__generator.next()
        i, j = 0, 0

        acc = self.__get_p(i=i)
        while x > acc:
            i += 1
            acc += self.__get_p(i=i)

        norm = self.__get_p(i=i)
        acc = self.__get_p(i, j) / norm
        
        while y > acc:
            j+=1
            acc += self.__get_p(i, j) / norm

        return i, j

    
    def get_enumerate(self, count):
        for _ in range(count):
            yield self.next()


if __name__ == '__main__':
    values = list(zip(*Generato2D(DISTRIBUTION).get_enumerate(N)))
    weights = np.zeros_like(values[0]) + 1./N

    all_x = list(range(len(DISTRIBUTION[0])))
    
    all_y_theory = [sum(row[j] for row in DISTRIBUTION) for j in range(len(DISTRIBUTION[0]))]
    all_y = [values[0].count(x)/N for x in all_x] 

    plt.plot(all_x, all_y)
    plt.plot(all_x, all_y_theory)
    plt.show()

    all_y_theory = [sum(DISTRIBUTION[i]) for i in range(len(DISTRIBUTION))]
    all_y = [values[1].count(x)/N for x in all_x] 
    
    plt.plot(all_x[:-1], all_y[:-1])
    plt.plot(all_x[:-1], all_y_theory)
    plt.show()

    m1 = sum(values[0]) / len(values[0])
    m2 = sum(values[1]) / len(values[1])
    print(f'M1) {M1_THEORY} ~ {m1}')
    print(f'M2) {M2_THEORY} ~ {m2}')

    d1 = sum((m1-x)**2 for x in values[0])/(len(values[0]) - 1)
    d2 = sum((m2-y)**2 for y in values[1])/(len(values[1]) - 1)
    print(f'D1) {D1_THEORY} ~ {d1}')
    print(f'D2) {D2_THEORY} ~ {d2}')

    cov = sum(val[0]*val[1] - m1*m2 for val in values) / N
    cor = sum((val[0]-m1)*(val[1]-m2) for val in values) / N / sqrt(d1*d2)


    print(f'cov) {cov}')
    print(f'cor) {cor} ~ {COR}')
