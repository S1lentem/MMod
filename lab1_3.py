from generators import DiscreteRandomGenerator2

import matplotlib.pyplot as plt

DISTRIBUTION = [(1, 0.25), (3, 0.25), (5, 0.1), (6, 0.1),
                (7, 0.2), (8, 0.05), (10, 0.05)]

COUNT = 100_000

if __name__ == '__main__':
    generator = DiscreteRandomGenerator2(DISTRIBUTION)

    values = list(generator.get_iterator(COUNT))

    plt.hist(values)
    plt.show()

    M = sum(values) / COUNT
    D = sum(value**2 - M**2 for value in values) / COUNT

    theory_m = sum(i[0]*i[1] for i in DISTRIBUTION)
    theory_d = sum(i[1]*(i[0] - theory_m)**2 for i in DISTRIBUTION)
    print(M, theory_m)
    print(D, theory_d)

    step = 2
    pair_sum = sum(values[i]*values[i+step] for i in range(len(values)-step))
    R = 12/(COUNT - step)*pair_sum
    print(R)