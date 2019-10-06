import numpy as np
import matplotlib.pyplot as plt
import scipy.special as scp_spec
import scipy as scp
import scipy.stats as scp_stat
import statistics as stat
import pynverse
import collections
import itertools
import random

class MCmethod_BRV_Generator():

    def __init__(self, start = 65539, multiplier = 65539, congruate_value = 2147483648):
        self.start = start
        self.multiplier = multiplier
        self.congruate_value = congruate_value
        self.current = start

    def generate_next(self):
        self.current = (self.multiplier * self.current) % self.congruate_value
        return self.current / self.congruate_value

class MMmethod_BRV_Generator():

    def __init__(self, 
                 #first_rv = MCmethod_BRV_Generator(congruate_value = 2147483648), 
                 first_rv = MCmethod_BRV_Generator(start = 65539, multiplier = 16807, congruate_value = 2147483647), 
                 second_rv = MCmethod_BRV_Generator(congruate_value = 2147483647), 
                 size = 256):
        self.size = size
        self.first_rv = first_rv
        self.second_rv = second_rv
        self.table = self._create_table(first_rv, size)

    def _create_table(self, rv, size):
        table = []
        for i in range(size):
            table.append(rv.generate_next())
        return table

    def generate_next(self):
        position = int(self.second_rv.generate_next() * self.size)
        result = self.table[position]
        self.table[position] = self.first_rv.generate_next()
        return result

def show_brv_generators_diff(size, *generators):
    for i, generator in enumerate(generators):
        
        value = 1 / len(generators) * i
        points = []
        value_list = []

        for i in range(0, size):
            points.append(generator.generate_next())
            value_list.append(value)

        plt.plot(points, value_list, 'ro')
    
    plt.show()

# compare uniform distribution with specific size
# mc_brv_generator = MCmethod_BRV_Generator()
# mm_brv_generator = MMmethod_BRV_Generator()
# show_brv_generators_diff(100, mc_brv_generator, mm_brv_generator)

class KoshiDistribution:

    left_limit = -5
    right_limit = 5

    params_count = 0

    @classmethod
    def get_func(cls, x):
        return 1 / np.pi * np.arctan(x) + 0.5

    @classmethod
    def get_density(cls, x):
        return 1 / np.pi * (1 / (x**2 + 1))

    @classmethod
    def get_inv_func(cls, x):
        return np.tan(np.pi * (x - 0.5))

    @classmethod
    def get_mean(cls):
        return 0

    @classmethod
    def get_dispersion(cls):
        return 0

class UniformDistribution:

    left_limit = 0
    right_limit = 1

    a = 0
    b = 1
    params_count = 2

    @classmethod
    def get_func(cls, x):
        if x < cls.a:
            return 0
        if x > cls.b:
            return 1
        return (x - cls.a) / (cls.b - cls.a)

    @classmethod
    def get_density(cls, x):        
        if x < cls.a:
            return 0
        if x > cls.b:
            return 0
        return 1 / (cls.b - cls.a)

    @classmethod
    def get_inv_func(cls, x):
        return x * (cls.b - cls.a) + cls.a

    @classmethod
    def get_mean(cls):
        return (cls.a + cls.b) / 2

    @classmethod
    def get_dispersion(cls):
        return (cls.b - cls.a)**2 / 12

class GammaDistribution:

    left_limit = 0
    right_limit = 20

    k = 2
    tetta = 2
    params_count = 2

    @classmethod
    def get_func(cls, x):
        return scp_spec.gammainc(cls.k, x / cls.tetta)

    @classmethod
    def get_density(cls, x):        
        return x**(cls.k-1) * (np.exp(-x / cls.tetta) / cls.tetta ** cls.k * scp_spec.gamma(cls.k))

    @classmethod
    def get_inv_func(cls, x):
        return scp_spec.gammaincinv(cls.k, x) * cls.tetta

    @classmethod
    def get_mean(cls):
        return cls.k * cls.tetta

    @classmethod
    def get_dispersion(cls):
        return cls.k * cls.tetta**2

def generate_crv(count, DistributionCls, BrvGeneratorCls):
    brv_generator = BrvGeneratorCls()
    rv_list = []
    for i in range(count):
        rv_F = brv_generator.generate_next()
        rv = DistributionCls.get_inv_func(rv_F)
        if DistributionCls.left_limit < rv < DistributionCls.right_limit:
            rv_list.append(rv)

    return rv_list

def compare_crv_function(rv_list, rv_F_list, gen_range, gen_F_list):
    plt.plot(rv_list, rv_F_list, 'ro')
    plt.plot(gen_range, gen_F_list, 'b')
    plt.show()

def create_histogram(bins_count, rv_list, not_close=False):
    hist = plt.hist(rv_list, bins = bins_count, density=True)
    if not_close: return hist

    plt.close()
    return hist

def compare_crv_density(bins_count, rv_list, gen_range, gen_density):
    create_histogram(bins_count, rv_list, True)
    plt.plot(gen_range, gen_density, 'r')
    plt.show()

def get_rv_mean(rv_list):
    return stat.mean(rv_list)

def get_rv_dispersion(rv_list):
    return stat.variance(rv_list)

def get_self_rv_dispersion(rv_list, mean):
    dispersion = 0
    for rv in rv_list:
        dispersion += (rv - mean) ** 2
    return dispersion / (len(rv_list) - 1)

def get_self_derv_mean(drv_map):
    p = 0
    s = 0
    for x, y in drv_map.items():
        s += x  * y
        p += y
    return s / p

def get_self_derv_dispersion(drv_map, rv_mean):
    rv_dispersion = 0
    for x, y in drv_map.items():
        rv_dispersion += (x - rv_mean) ** 2 * y
    return rv_dispersion

def compare_rv_mean(DistributionCls, rv_mean):
    print("Mean: theoratical - {}, generated - {}".format(DistributionCls.get_mean(), rv_mean))

def compare_rv_dispersion(DistributionCls, rv_dispersion):
    print("Dispersion: theoratical - {}, generated - {}".format(DistributionCls.get_dispersion(), rv_dispersion))

def compare_params_estimate(DistributionCls, rv_mean, rv_dispersion):
    scale_estimate = rv_dispersion / rv_mean
    shape_estimate = rv_mean / scale_estimate
    print("Shape param: actual - {}, estimate - {}".format(DistributionCls.k, shape_estimate))
    print("Scale param: actual - {}, estimate - {}".format(DistributionCls.tetta, scale_estimate))

def get_rv_mean_interval(confidence, rv_list, rv_dispersion):
    return np.sqrt(rv_dispersion) / np.sqrt(len(rv_list) - 1) * scp_stat.norm.ppf(confidence)

def show_mean_interval(rv_mean, mean_interval):    
    print("Mean interval: {} - {}".format(rv_mean - mean_interval, rv_mean + mean_interval))

def show_mean_data_on_function_plot(gen_range, rv_list, rv_F_list, DistributionCls, rv_mean, mean_interval):
    plt.plot(rv_list, rv_F_list, 'ro')
    plt.plot(gen_range, DistributionCls.get_func(gen_range), 'b')
    plt.axvline(x = rv_mean)
    plt.text(rv_mean + 0.1, 0, 'Estimate', rotation=90)
    plt.axvline(x = DistributionCls.get_mean())
    plt.text(DistributionCls.get_mean() + 0.1, 0, 'Theoratical', rotation=90)
    plt.axvline(x = rv_mean - mean_interval, c='g')
    plt.axvline(x = rv_mean + mean_interval, c='g')
    plt.show()

def get_rv_dispersion_intervals(confidence, rv_list, rv_dispersion):
    dispersion_left_interval = len(rv_list) * rv_dispersion / scp_stat.chi2.isf((1 - confidence) / 2, len(rv_list) - 1)
    dispersion_right_interval = len(rv_list) * rv_dispersion / scp_stat.chi2.isf((1 + confidence) / 2, len(rv_list) - 1)
    return (dispersion_left_interval, dispersion_right_interval)

def show_dispersion_intervals(dispersion_left_interval, dispersion_right_interval):
    print("Dispersion interval: {} - {}".format(dispersion_left_interval, dispersion_right_interval))

def show_dispersion_data_on_histogram_plot(bins_count, rv_list, gen_range, DistributionCls, dispersion_left_interval, dispersion_right_interval):
    create_histogram(bins_count, rv_list, True)
    plt.plot(gen_range, DistributionCls.get_density(gen_range), 'r')
    plt.axvline(x = DistributionCls.get_dispersion(), c='y')
    plt.axvline(x = dispersion_left_interval, c='g')
    plt.axvline(x = dispersion_right_interval, c='g')
    plt.show()












class BinomialDistribution:

    n = 20
    p = 0.7

    left_limit = 0
    right_limit = 30
    params_count = 2

    @classmethod
    def _binomial_koef(cls, n, k):
        return np.math.factorial(n) / (np.math.factorial(n - k) * np.math.factorial(k))

    @classmethod
    def get_func(cls, y):
        f = 0
        for k in range(int(y)):
            if k > cls.n:
                break
            f += cls._binomial_koef(cls.n, k) * cls.p ** k * (1 - cls.p) ** (cls.n - k)
        return f

    @classmethod
    def get_density(cls, k):
        if (k > cls.n):
            return 0 
        return cls._binomial_koef(cls.n, k) * cls.p ** k * (1 - cls.p) ** (cls.n - k) 

    @classmethod
    def get_mean(cls):
        return cls.n * cls.p

    @classmethod
    def get_dispersion(cls):
        return cls.n * cls.p * (1 - cls.p)

class GeometricDistribution:

    p = 0.5

    left_limit = 0
    right_limit = 10
    params_count = 1

    @classmethod
    def get_func(cls, n):
        return 1 - (1 - cls.p) ** (n + 1)

    @classmethod
    def get_density(cls, n):
        return (1 - cls.p) ** n * cls.p

    @classmethod
    def get_mean(cls):
        return (1 - cls.p) / cls.p

    @classmethod
    def get_dispersion(cls):
        return (1 - cls.p) / cls.p**2


class PuassonDistribution:

    lamb = 1

    left_limit = 0
    right_limit = 20
    params_count = 1

    @classmethod
    def get_func(cls, k):
        return (scp_spec.gammaincc(k + 1, cls.lamb) * scp_spec.gamma(k + 1)) / np.math.factorial(k)

    @classmethod
    def get_density(cls, k):
        return np.exp(-cls.lamb) * cls.lamb ** k / np.math.factorial(k)

    @classmethod
    def get_mean(cls):
        return cls.lamb

    @classmethod
    def get_dispersion(cls):
        return cls.lamb

def get_theor_density_map(gen_range, DistributionCls):
    distribution_map = {}
    for item_in_range in gen_range:
        distribution_map[item_in_range] = DistributionCls.get_density(item_in_range)
    return distribution_map

def get_theor_function_map(gen_range, DistributionCls):
    distribution_map = {}
    for item_in_range in gen_range:
        distribution_map[item_in_range] = DistributionCls.get_func(item_in_range)
    return distribution_map

def build_help_vector(theor_density_list):
    help_vector_list = [0]
    for i in range(len(theor_density_list)):
        q = 0
        for j in range(i+1):
            q += theor_density_list[j]
        help_vector_list.append(q)

    return help_vector_list

def generate_drv(count, theor_density_map, BrvGeneratorCls, DistributionCls):
    theor_density_list = list(theor_density_map.values())
    theor_x_list = list(theor_density_map.keys())
    help_vector_list = build_help_vector(theor_density_list)

    brv_generator = BrvGeneratorCls()

    rv_map = {}
    rv_frequency_map = {}
    for i in range(count):
        generated_value = brv_generator.generate_next()

        for j in range(1, len(help_vector_list)):
            if help_vector_list[j - 1] < generated_value < help_vector_list[j]:
                
                rv = theor_x_list[j-1]
                freq = rv_frequency_map.get(rv, 0) + 1
                rv_frequency_map[rv] = freq
                rv_map[rv] = freq / count

                break

    return collections.OrderedDict(sorted(rv_map.items()))

def show_rv_density(theor_density_map, rv_list_map):
    plt.plot(list(theor_density_map.keys()), list(theor_density_map.values()), 'ro')
    plt.plot(list(rv_list_map.keys()), list(rv_list_map.values()), 'bo')
    plt.show()

def get_drv_function(rv_list_map):
    rv_function_list = []
    rv_density_list = list(rv_list_map.values())
    for i in range(len(rv_list_map)):
        rv_f = 0
        for j in range(i+1):
            rv_f += rv_density_list[j]
        rv_function_list.append(rv_f)
    return rv_function_list

def show_rv_function(theor_function_map, rv_list, rv_function_list):
    plt.plot(list(theor_function_map.keys()), list(theor_function_map.values()), 'ro')
    plt.plot(rv_list, rv_function_list, 'bo')
    plt.show()

def show_drv_poligon(theor_density_map, rv_list_map):
    plt.plot(list(theor_density_map.keys()), list(theor_density_map.values()), 'ro')
    plt.plot(list(rv_list_map.keys()), list(rv_list_map.values()), 'b')
    plt.show()





# def check_independence_with_chi2(rv_list, intervals_count):
#     first_half_rv = []
#     second_half_rv = []
#     for rv in rv_list:
#         if rv < 0.5:
#             first_half_rv.append(rv)
#         else:
#             second_half_rv.append(rv - 0.5)

#     m = min(len(first_half_rv), len(second_half_rv))
#     first_half_rv = first_half_rv[:m]
#     second_half_rv = second_half_rv[:m]

#     intervals = np.linspace(0, 0.5, intervals_count + 1)

#     frequency_table = [[0 for j in range(intervals_count)] for i in range(intervals_count)]
#     frequency_first_list = [0 for i in range(intervals_count)]
#     frequency_second_list = [0 for i in range(intervals_count)]

#     for i in range(m):
#         in_first_index = None
#         for j in range(1, intervals_count + 1):
#             if intervals[j-1] < first_half_rv[i] < intervals[j]:
#                 frequency_first_list[j-1] += 1
#                 in_first_index = j-1
#                 break
#         in_second_index = None
#         for j in range(1, intervals_count + 1):
#             if intervals[j-1] < second_half_rv[i] < intervals[j]:
#                 frequency_second_list[j-1] += 1
#                 in_second_index = j-1
#                 break
#         if in_first_index is not None and in_second_index is not None:
#             frequency_table[in_first_index][in_second_index] += 1

#     chi2_estimate = 0
#     for i in range(intervals_count):
#         for j in range(intervals_count):
#             if frequency_first_list[i] * frequency_second_list[j] == 0:
#                 continue
#             chi2_estimate += (frequency_table[i][j] - (frequency_first_list[i] * frequency_second_list[j]) / m) ** 2 / (frequency_first_list[i] * frequency_second_list[j])
#     chi2_estimate *= m
#     print(chi2_estimate)
#     print(scp_stat.chi2.ppf(0.95, (intervals_count-1)**2))

def check_half_independence_with_chi2(brv_generator, count, intervals_count):
    def split():
        # rv_list = [brv_generator.generate_next() for i in range(count)]
        # rv_list.sort()
        first_half_rv = []
        second_half_rv = []
        for i in range(count):
            rv = brv_generator.generate_next()
            if rv < 0.5:
                first_half_rv.append(rv)
            else:
                second_half_rv.append(rv - 0.5)
        return first_half_rv, second_half_rv
            
    print('Half independence')
    check_independence_with_chi2(intervals_count, split, 0.5)

def check_every_second_independence_with_chi2(brv_generator, count, intervals_count):
    def split():
        first_half_rv = []
        second_half_rv = []
        k = 0
        for i in range(count):
            rv = brv_generator.generate_next()
            if k == 0:
                k = 1
                first_half_rv.append(rv)
            else:
                k = 0
                second_half_rv.append(rv)
        return first_half_rv, second_half_rv

    print('Every second')
    check_independence_with_chi2(intervals_count, split, 1)

def check_two_big_independence_with_chi2(brv_generator, count, intervals_count):
    def split():
        first_half_rv = []
        second_half_rv = []
        for i in range(count):
            first_half_rv.append(brv_generator.generate_next())
        for i in range(count):
            second_half_rv.append(brv_generator.generate_next())
        return first_half_rv, second_half_rv

    print('Two big')
    check_independence_with_chi2(intervals_count, split, 1)

def check_independence_with_chi2(intervals_count, separation_func, right_limit):
    first_half_rv, second_half_rv = separation_func()

    m = min(len(first_half_rv), len(second_half_rv))
    first_half_rv = first_half_rv[:m]
    second_half_rv = second_half_rv[:m]

    # plt.plot(first_half_rv, second_half_rv, 'bo')
    # plt.show()

    intervals = np.linspace(0, right_limit, intervals_count + 1)

    frequency_table = [[0 for j in range(intervals_count)] for i in range(intervals_count)]
    frequency_first_list = [0 for i in range(intervals_count)]
    frequency_second_list = [0 for i in range(intervals_count)]

    for i in range(m):
        in_first_index = None
        for j in range(1, intervals_count + 1):
            if intervals[j-1] < first_half_rv[i] < intervals[j]:
                frequency_first_list[j-1] += 1
                in_first_index = j-1
                break
        in_second_index = None
        for j in range(1, intervals_count + 1):
            if intervals[j-1] < second_half_rv[i] < intervals[j]:
                frequency_second_list[j-1] += 1
                in_second_index = j-1
                break
        if in_first_index is not None and in_second_index is not None:
            frequency_table[in_first_index][in_second_index] += 1

    chi2_estimate = 0
    for i in range(intervals_count):
        for j in range(intervals_count):
            if frequency_first_list[i] * frequency_second_list[j] == 0:
                continue
            chi2_estimate += (frequency_table[i][j] - (frequency_first_list[i] * frequency_second_list[j]) / m) ** 2 / (frequency_first_list[i] * frequency_second_list[j])
    chi2_estimate *= m

    print(chi2_estimate)
    print(scp_stat.chi2.ppf(0.95, (intervals_count-1)**2))

def chi_square_estimate_for_crv(rv_list, DistributionCls, intervals_count = 50, confedence = 0.99):
    intervals = np.linspace(DistributionCls.left_limit, DistributionCls.right_limit, intervals_count + 1)

    n = len(rv_list)

    frequency_list = [0 for i in range(intervals_count)]
    for rv in rv_list:
        for j in range(1, intervals_count + 1):
            if intervals[j-1] < rv < intervals[j]:
                frequency_list[j-1] += 1
                break
            
    frequency_list = [freq / n for freq in frequency_list]

    chi_square = 0
    for i in range(intervals_count):
        theor_density = DistributionCls.get_func(intervals[i+1]) - DistributionCls.get_func(intervals[i])
        chi_square += (frequency_list[i] - theor_density) ** 2 / theor_density
    chi_square *= n

    print('Chi-square:', chi_square)
    print('Test chi-square value:', scp_stat.chi2.ppf(confedence, intervals_count - 1 - DistributionCls.params_count))

def chi_square_estimate_for_drv(count, rv_list, rv_density, DistributionCls, confedence = 0.99):

    n = len(rv_list)

    chi_square = 0
    for i in range(n):
        theor_density = DistributionCls.get_density(rv_list[i])
        chi_square += (rv_density[i] - theor_density) ** 2 / theor_density
    chi_square *= count

    print('Chi-square:', chi_square)
    print('Test chi-square value:', scp_stat.chi2.ppf(confedence, n - 1 - DistributionCls.params_count))


# check problem with function shift
def kalmagorov_estimate_for_crv(rv_list, rv_function_list, DistributionCls, confedence = 0.99):

    max_shift = 0
    for i in range(len(rv_list)):
        theor = DistributionCls.get_func(rv_list[i])
        empir = rv_function_list[i]
        if abs(theor - empir) > max_shift:
            max_shift = abs(theor - empir)
    max_shift *= np.sqrt(len(rv_list)) * max_shift

    print('Kalmogorov:', max_shift)
    print('Test kalmogorov value:', scp_spec.kolmogorov(confedence))


def run_test_brv(count, BrvGeneratorCls, confidence = 0.99):
    brv_generator = BrvGeneratorCls()

    brv_list = []
    for i in range(count):
        brv_list.append(brv_generator.generate_next())

    # brv_list = []
    # for i in range(count):
    #     brv_list.append(random.random())

    brv_map = {}
    for brv in brv_list:
        brv_map[brv] = 1 / count
    brv_map = collections.OrderedDict(sorted(brv_map.items()))

    original_brv_list = list(brv_list)
    brv_list = list(brv_map.keys())
    brv_density_list = list(brv_map.values())
    brv_function_list = get_drv_function(brv_map)

    mean = 0
    for brv in brv_list:
        mean += brv
    mean /= len(brv_list)

    print('Mean for brv: theor - {}, actual - {}'.format(1/2, mean))

    dispersion = 0
    for brv in brv_list:
        dispersion += (brv - mean) ** 2
    dispersion /= len(brv_list) - 1

    print('Dispersion for brv: theor - {}, actual - {}'.format(1/12, dispersion))

    mean_interval = get_rv_mean_interval(confidence, brv_list, dispersion)
    show_mean_interval(mean, mean_interval)

    dispersion_left_interval, dispersion_right_interval = get_rv_dispersion_intervals(confidence, brv_list, dispersion)
    show_dispersion_intervals(dispersion_left_interval, dispersion_right_interval)

    hist_count = int(count ** 0.5) if count <= 100 else int(4 * np.log(count))
    hist = create_histogram(hist_count, brv_list, True)
    plt.show()

    plt.plot(brv_list, brv_function_list, 'ro')
    plt.show()


    print('')

    freq_list = [0 for i in range(hist_count-1)]
    for brv in brv_list:
        for i in range(1, hist_count):
            if hist[1][i-1] < brv < hist[1][i]:
                freq_list[i-1] += 1
    freq_list = [f / count for f in freq_list]
    average_freq = 0
    for freq in freq_list:
        average_freq += freq
    average_freq /= len(freq_list)
    print('Average frequency: ', average_freq)
    print('Theor probability:', 1 / hist_count)

    # freq = [0 for i in range(hist_count-1)]
    # comb = itertools.permutations(brv_list, 2)
    # comb_c = 0
    # for perm in comb:
    #     comb_c += 1
    #     for i in range(1, hist_count):
    #         if hist[1][i-1] < perm[0] < hist[1][i] and hist[1][i-1] < perm[1] < hist[1][i]:
    #             freq[i-1] += 11
    # freq = [f / comb_c for f in freq]
    # av_freq = 0
    # for f in freq:
    #     av_freq += f
    # av_freq /= len(freq)
    # print('Frequency', av_freq)
    # print('Theor probability:', 1 / hist_count**2)

    print('')
    
    chi_square_estimate_for_crv(brv_list, UniformDistribution, hist_count)

    # step = 1
    # brv_sum = 0
    # for i in range(count - step):
    #     brv_sum += original_brv_list[i] * original_brv_list[i + step]
    # r_estimate = 12 * 1 / (count - step) * brv_sum - 3
    # print('Independence estimate:', r_estimate)

    print('')

    print('Kovariation')
    step = 1
    brv_sum = 0
    for i in range(count - step):
        brv_sum += original_brv_list[i] * original_brv_list[i + step]
    r_estimate = 1 / (count - step - 1) * brv_sum - count / (count - 1) * mean**2
    print(abs(r_estimate))
    print(scp_stat.norm.ppf(0.99) / 12 / np.sqrt(count - 1))

    print('')

    print('\nIndependence\n')
    check_half_independence_with_chi2(BrvGeneratorCls(), count, hist_count)
    check_every_second_independence_with_chi2(BrvGeneratorCls(), count, hist_count)
    check_two_big_independence_with_chi2(BrvGeneratorCls(), count, hist_count)




def run_crv_analyze(count, BrvGeneratorCls, DistributionCls, confidence = 0.99):
    gen_range = np.arange(DistributionCls.left_limit, DistributionCls.right_limit, 0.1)

    rv_list = generate_crv(count, DistributionCls, BrvGeneratorCls)
    
    rv_map = {}
    for rv in rv_list:
        rv_map[rv] = 1 / count
    
    rv_map = collections.OrderedDict(sorted(rv_map.items()))
    
    original_rv_list = list(rv_list)
    rv_list = list(rv_map.keys())
    rv_density_list = list(rv_map.values())
    rv_function_list = get_drv_function(rv_map)

    # show how function is filled
    compare_crv_function(rv_list, rv_function_list, gen_range, DistributionCls.get_func(gen_range))

    # show how density is filled
    intervals_count = int(count ** 0.5) if count <= 100 else int(4 * np.log(count))
    hist = create_histogram(intervals_count, rv_list)
    compare_crv_density(intervals_count, rv_list, gen_range, DistributionCls.get_density(gen_range))

    print('\nPoints\n')

    rv_mean = get_rv_mean(rv_list)
    rv_dispersion = get_rv_dispersion(rv_list) # состоятельная и несмещенная

    compare_rv_mean(DistributionCls, rv_mean)
    compare_rv_dispersion(DistributionCls, rv_dispersion)
    compare_params_estimate(DistributionCls, rv_mean, rv_dispersion)

    print('\nIntervals\n')

    mean_interval = get_rv_mean_interval(confidence, rv_list, rv_dispersion)
    show_mean_interval(rv_mean, mean_interval)
    #show_mean_data_on_function_plot(gen_range, rv_list, rv_F_list, DistributionCls, rv_mean, mean_interval)

    dispersion_left_interval, dispersion_right_interval = get_rv_dispersion_intervals(confidence, rv_list, rv_dispersion)
    show_dispersion_intervals(dispersion_left_interval, dispersion_right_interval)
    #show_dispersion_data_on_histogram_plot(hist_bins, rv_list, gen_range, DistributionCls, dispersion_left_interval, dispersion_right_interval)

    print('\nTest\n')

    chi_square_estimate_for_crv(rv_list, DistributionCls, intervals_count)
    kalmagorov_estimate_for_crv(rv_list, rv_function_list, DistributionCls)

def run_drv_analyze(count, BrvGeneratorCls, DistributionCls, confidence = 0.99):
    gen_range = np.arange(DistributionCls.left_limit, DistributionCls.right_limit, 1)

    theor_density_map = get_theor_density_map(gen_range, DistributionCls)
    theor_function_map = get_theor_function_map(gen_range, DistributionCls)

    rv_map = generate_drv(count, theor_density_map, BrvGeneratorCls, DistributionCls)

    rv_list = list(rv_map.keys())
    rv_density_list = list(rv_map.values())
    rv_function_list = get_drv_function(rv_map)

    if theor_function_map[0] == 0:
        rv_function_list.insert(0, 0)
        rv_function_list = rv_function_list[:-1]        

    show_rv_function(theor_function_map, rv_list, rv_function_list)
    show_rv_density(theor_density_map, rv_map)
    show_drv_poligon(theor_density_map, rv_map)

    print('\nPoints\n')

    rv_mean = get_self_derv_mean(rv_map)
    rv_dispersion = get_self_derv_dispersion(rv_map, rv_mean)

    compare_rv_mean(DistributionCls, rv_mean)
    compare_rv_dispersion(DistributionCls, rv_dispersion)

    print('\nIntervals\n')

    mean_interval = get_rv_mean_interval(confidence, rv_list, rv_dispersion)
    show_mean_interval(rv_mean, mean_interval)

    dispersion_left_interval, dispersion_right_interval = get_rv_dispersion_intervals(confidence, rv_list, rv_dispersion)
    show_dispersion_intervals(dispersion_left_interval, dispersion_right_interval)

    print('\nTest\n')

    kalmagorov_estimate_for_crv(rv_list, rv_function_list, DistributionCls)
    chi_square_estimate_for_drv(count, rv_list, rv_density_list, DistributionCls)
        
# 2**13 (xn-1 + xn-2 + xn-3) mod 2**32 - 5 период 2**96

def test_generators():

    class Test_Generator():

        def __init__(self, start = 11, multiplier = 101, c = 9, congruate_value = 103):
            self.start = start
            self.c = c
            self.multiplier = multiplier
            self.congruate_value = congruate_value
            self.current = start

        def generate_next(self):
            self.current = (self.multiplier * self.current + self.c) % self.congruate_value
            return self.current / self.congruate_value

    class Test_MM_Generator():

        def __init__(self, 
                    first_rv = MCmethod_BRV_Generator(congruate_value = 2147483648), 
                    second_rv = MCmethod_BRV_Generator(congruate_value = 2147483647), 
                    size = 256):
            self.size = size
            self.first_rv = first_rv
            self.second_rv = second_rv
            self.table = self._create_table(first_rv, size)

        def _create_table(self, rv, size):
            table = []
            for i in range(size):
                table.append(rv.generate_next())
            return table

        def generate_next(self):
            position = int(self.second_rv.generate_next() * self.size)
            result = self.table[position]
            self.table[position] = self.first_rv.generate_next()
            return result

    generator1 = Test_Generator(start = 11, multiplier = 34, c = 13, congruate_value = 99)
    generator2 = Test_Generator(start = 9, multiplier = 11, c = 9, congruate_value = 50)

    def get_period(generator):
        first = generator.generate_next()
        new = generator.generate_next()
        c = 1
        while new != first:
            c += 1
            new = generator.generate_next()
            print(new)
        return c

    first_period = get_period(generator1)
    # print('Congruent 1:', first_period)
    # second_period = get_period(generator2)
    # print('Congruent 2:', second_period)

    # generator1 = Test_Generator(start = 11, multiplier = 34, c = 13, congruate_value = 99)
    # generator2 = Test_Generator(start = 9, multiplier = 11, c = 9, congruate_value = 50)
    # generator_MM = Test_MM_Generator(first_rv = generator1, 
    #                 second_rv = generator2, 
    #                 size = 99)
    # mm_period = get_period(generator_MM)
    # print('Congruent MM:', mm_period)


# test_generators()

#print('\n\nBRV\n\n')
 
# run_test_brv(10000, MMmethod_BRV_Generator)

# plt.close()

# print('\n\nCRV\n\n')

#run_crv_analyze(1000, MMmethod_BRV_Generator, GammaDistribution)

# plt.close()

# print('\n\nDRV\n\n')

run_drv_analyze(1000, MMmethod_BRV_Generator, BinomialDistribution)