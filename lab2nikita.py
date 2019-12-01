import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from matplotlib import cm
import seaborn as sns
from pandas import DataFrame
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D

int32_t_MAX = 2147483648
uint16_t_MAX = 65539
prog_seed = uint16_t_MAX

    
def density_of_xy(x, y):
    return np.sin(y + x) / 2

def density_of_x(x_basic):
    return (np.cos(x_basic) + np.sin(x_basic)) / 2

def function_of_x(x):
    return (np.sin(x) - np.cos(x)) / 2 + 0.5

def inverse_function_of_x(x):
    return np.arccos(0.5 * (np.sqrt(- 4 * (x ** 2) + 4 * x + 1) - 2 * x + 1))

def density_of_y_x(y_basic, x_basic):
    return np.sin(y_basic + x_basic) / (np.cos(x_basic) + np.sin(x_basic))

def function_of_y_x(y, x):
    return (np.cos(x) - np.cos(y + x))/(np.sin(x) + np.cos(x))

def inverse_function_of_y_x(y, x):
    return -x + np.arccos(y * (-np.sin(x)) - y * np.cos(x) + np.cos(x))


class CongruentGenerator:
    """
        This class is designed to generate basic random values using 
        multiply-congruent method
    """
    def __init__(self, seed, multiplier, divider):
        def condition_valid():
            is_all_integer = (isinstance(seed, int) or seed.is_integer()) and (isinstance(multiplier, int) or multiplier.is_integer()) and (isinstance(divider, int) or divider.is_integer())
            is_a_in_range = seed > 0 and seed < divider
            is_multiplier_valid = multiplier < divider
            return is_all_integer and is_a_in_range and is_multiplier_valid
        
        if not condition_valid():
            raise ValueError("Invalid start data for BRV generator")
    
        self.multiplier = multiplier
        self.divider = divider
        self.a_star = seed
        
    def generate_value(self):
        next_a_star = (self.multiplier * self.a_star) % self.divider
        generated_value = self.a_star / self.divider
        self.a_star = next_a_star
        return generated_value


class MclarenMarclesGenerator:
    """
        This class is designed to generate basic random values using 
        MclarenMarcles method, with the support of two CongruentGenerators
    """
    def __init__(self, seed, multiplier, divider):
        def condition_valid():
            is_all_integer = (isinstance(seed, int) or seed.is_integer()) and (isinstance(multiplier, int) or multiplier.is_integer()) and (isinstance(divider, int) or divider.is_integer())
            is_a_in_range = seed > 0 and seed < divider
            is_multiplier_valid = multiplier < divider
            return is_all_integer and is_a_in_range and is_multiplier_valid
        
        if not condition_valid():
            raise ValueError("Invalid start data for BRV generator")
    
        self.multiplier = multiplier
        self.divider = divider
        self.a_star = seed
        
        mid_gen = CongruentGenerator(seed, multiplier, divider)
        
        self.b_cache, self.c_cache, self.v_cache = [], [], []
        self.b_generator = CongruentGenerator(int(mid_gen.generate_value() * (divider - 1)), multiplier, divider)
        self.c_generator = CongruentGenerator(int(mid_gen.generate_value() * (divider - 1)), multiplier, divider)
        
    def generate_value(self):
        self.b_cache.append(self.b_generator.generate_value())
        self.c_cache.append(self.c_generator.generate_value())
        self.v_cache.append(self.b_cache[-1]) 
        index = int(self.c_cache[-1] * len(self.v_cache))
        generated_value = self.v_cache[index]
        self.v_cache[index] = self.b_cache[-1]
        return generated_value

###############################################################################
#                                  TASK A                                     #
###############################################################################
print('TASK A')

congruentGenerator = CongruentGenerator(prog_seed, uint16_t_MAX, int32_t_MAX)


class GeneratorCRV2D:
    def __init__(self):
        self.generator = CongruentGenerator(prog_seed, uint16_t_MAX, int32_t_MAX)
    
    def generateValue(self):
        x = self.generator.generate_value()
        y = self.generator.generate_value()
        X = inverse_function_of_x(x)
        Y = inverse_function_of_y_x(y, X)
        return X, Y


# simple density and function plots for check
        
x = np.linspace(0, np.pi / 2, 1000)
plt.plot(x, list(map(density_of_x, x)), color='g')
plt.ylabel('x density: ')
plt.show()

x = np.linspace(0, np.pi / 2, 1000)
plt.plot(x, list(map(function_of_x, x)), color='g')
plt.ylabel('x function: ')
plt.show()

x = np.linspace(0, 1, 1000)
plt.plot(x, list(map(inverse_function_of_x, x)), color='g')
plt.ylabel('x inverse function: ')
plt.show()

x = np.linspace(0, np.pi / 2, 1000)
plt.plot(x, list(map(density_of_y_x, x, [np.pi/2 for x in x])), color='g')
plt.ylabel('y_x density (x = pi/2): ')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.linspace(0, np.pi / 2, 50)
Y = np.linspace(0, np.pi / 2, 50)
X, Y = np.meshgrid(X, Y)
Z = density_of_y_x(Y, X)
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_zlim(0, 1.0)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.linspace(0, np.pi / 2, 50)
Y = np.linspace(0, np.pi / 2, 50)
X, Y = np.meshgrid(X, Y)
Z = function_of_y_x(Y, X)
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_zlim(0.0, 1.0)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


x = np.linspace(0, np.pi / 2, 1000)
plt.plot(x, list(map(function_of_y_x, x, [np.pi/2 for x in x])), color='g')
plt.ylabel('y|x function (x = pi/2): ')
plt.show()

x = np.linspace(0, 1, 1000)
plt.plot(x, list(map(inverse_function_of_y_x, x, [np.pi/2 for x in x])), color='g')
plt.ylabel('y|x inverse function (x = pi/2): ')
plt.show()


# generating 2d CRV and checking them
# TOOD: make larger
dataset_len = 10000
generator = GeneratorCRV2D()
dataset = [generator.generateValue() for _ in range(dataset_len)]
zipped_data = list(zip(*dataset))

sns.jointplot(x=zipped_data[0],
              y=zipped_data[1], 
              kind="kde", 
              space=0)
print('show')
plt.show()

dataset_len = 1000000
generator = GeneratorCRV2D()
dataset = [generator.generateValue() for _ in range(dataset_len)]
zipped_data = list(zip(*dataset))
frequency = defaultdict(int)
bin_len = 0.16
for x, y in dataset: 
    frequency[(int(x//bin_len), int(y//bin_len))]+=1/dataset_len
    
heat_map = []
previous_x = -1
for x, y in sorted(frequency.keys()):
    if previous_x != x:
        heat_map.append([])
        previous_x = x
    heat_map[x].append(frequency[(y, x)])

frameData = DataFrame(heat_map)
sns.heatmap(data=frameData)
plt.show()

sns.distplot(zipped_data[0], bins=10, kde=False, norm_hist=True)
x = np.linspace(0, np.pi / 2, 1000)
frameData = DataFrame(data={'x': x, 'y': [density_of_x(val) for val in x]})
sns.lineplot(x='x', y='y', data=frameData, palette="ch:2.5,.25")
plt.show()


sns.distplot([inverse_function_of_y_x(congruentGenerator.generate_value(), np.pi/2) for i in range(dataset_len)], 
              bins=10, kde=False, norm_hist=True)
x = np.linspace(0, np.pi / 2, 1000)
frameData = DataFrame(data={'x': x, 'y': [density_of_y_x(val, np.pi/2) for val in x]})
sns.lineplot(x='x', y='y', data=frameData, palette="ch:2.5,.25")
plt.show()

theoretical_math_e_x = np.pi / 4
sns.distplot(zipped_data[1], bins=10, kde=False, norm_hist=True)
x = np.linspace(0, np.pi / 2, 1000)
frameData = DataFrame(data={'x': x, 'y': [density_of_y_x(val, theoretical_math_e_x) for val in x]})
sns.lineplot(x='x', y='y', data=frameData, palette="ch:2.5,.25")
plt.show()

# Practical data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = zipped_data[0]
y = zipped_data[1]
hist, xedges, yedges = np.histogram2d(x, y, bins=32, range=[[0, np.pi / 2], [0, np.pi / 2]], density=True)

# Construct arrays for the anchor positions of the 16 bars.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the 16 bars.
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

plt.show()


# Theoretical data
fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.linspace(0, np.pi / 2, 50)
Y = np.linspace(0, np.pi / 2, 50)
X, Y = np.meshgrid(X, Y)
Z = density_of_xy(X, Y)
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

plt.show()

theoretical_math_e_x = np.pi / 4
theoretical_math_e_y = (1 / 2 * (np.pi - 2) * np.sin(theoretical_math_e_x) + np.cos(theoretical_math_e_x)) / (np.sin(theoretical_math_e_x) + np.cos(theoretical_math_e_x))

x = zipped_data[0]
y = zipped_data[1]

practical_math_e_x = sum(x) / len(x)
practical_math_e_y = sum(y) / len(y)

print('Math exp:')
print('X:')
print('theoretical_math_e_x:', theoretical_math_e_x)
print('practical_math_e_x:', practical_math_e_x)
print('Y:')
print('theoretical_math_e_y:', theoretical_math_e_y)
print('practical_math_e_y:', practical_math_e_y)

theoretical_dispersion_x = 1 / 16 * (-32 + 8 * np.pi + np.pi ** 2)
theoretical_dispersion_y = 1 / 16 * (-32 + 8 * np.pi + np.pi ** 2)
practical_dispersion_x = sum([(practical_math_e_x - item) ** 2 for item in x]) / (len(x) - 1)
practical_dispersion_y = sum([(practical_math_e_y - item) ** 2 for item in y]) / (len(y) - 1)

print('\nDispersion:')
print('X:')
print('theoretical_dispersion_x:', theoretical_dispersion_x)
print('practical_dispersion_x:', practical_dispersion_x)
print('Y:')
print('theoretical_dispersion_y:', theoretical_dispersion_y)
print('practical_dispersion_y:', practical_dispersion_y)

corellation_coeff_theoretical = ((np.pi - 2) / 2 - theoretical_math_e_x * theoretical_math_e_y) / np.sqrt(theoretical_dispersion_y * theoretical_dispersion_x)
corellation_coeff_practical = sum([(_x - practical_math_e_x) * (_y - practical_math_e_y) for _x, _y in dataset]) / dataset_len
corellation_coeff_practical /= (np.sqrt(practical_dispersion_x) * np.sqrt(practical_dispersion_y))

print('\nCorellation coeff:')
print('theoretical', corellation_coeff_theoretical)
print('practical', corellation_coeff_practical)

def check_intervals(non_offsetted_estimate_dispersion, math_estimate, real_dispersion, real_math_exp, dataset_len):
    gamma = 0.99
    math_exp_delta = (non_offsetted_estimate_dispersion ** 0.5) / (dataset_len - 1) ** 0.5 * scipy.stats.norm.ppf(gamma)
    
    print('math delta: ', math_exp_delta)
    math_lower_bound, math_upper_bound = real_math_exp - math_exp_delta, real_math_exp + math_exp_delta
    if math_estimate >= math_lower_bound and math_estimate < math_upper_bound:
        print('Math expectations interval check verdict: {} <= {} < {}, we are in this range'.format(math_lower_bound, math_estimate, math_upper_bound))   
    else:
        print('Math expectations interval check verdict: not {} <= {} < {}, we are not in this range'.format(math_lower_bound, math_estimate, math_upper_bound))

    dispersion_lower_bound = dataset_len * non_offsetted_estimate_dispersion / scipy.stats.chi2.isf((1 - gamma) / 2, dataset_len - 1)
    dispersion_upper_bound = dataset_len * non_offsetted_estimate_dispersion / scipy.stats.chi2.isf((1 + gamma) / 2, dataset_len - 1)
    
    if non_offsetted_estimate_dispersion >= dispersion_lower_bound and non_offsetted_estimate_dispersion < dispersion_upper_bound:
        print('Dispersion expectations interval check verdict: {} <= {} < {}, we are in this range'.format(dispersion_lower_bound, non_offsetted_estimate_dispersion, dispersion_upper_bound))   
    else:
        print('Dispersion expectations interval check verdict: not {} <= {} < {}, we are not in this range'.format(dispersion_lower_bound, non_offsetted_estimate_dispersion, dispersion_upper_bound))


def check_corellation_interval(theoretical_corellation, practical_corellation, dataset_len):
    gamma = 0.99
    z = 1 / 2 * np.log((1 + theoretical_corellation) / (1 - theoretical_corellation))
    delta_z = scipy.stats.norm.ppf(gamma) / np.sqrt(dataset_len - 3)
    z_lower, z_upper = z - delta_z, z + delta_z
    lower_bound = ((np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)) 
    upper_bound = ((np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1))
    
    if practical_corellation >= lower_bound and practical_corellation < upper_bound:
        print('Corellation expectations interval check verdict: {} <= {} < {}, we are in this range'.format(lower_bound, practical_corellation, upper_bound))   
    else:
        print('Corellation expectations interval check verdict: not {} <= {} < {}, we are not in this range'.format(lower_bound, practical_corellation, upper_bound))

print('\nX intervals')
check_intervals(practical_dispersion_x, practical_math_e_x, theoretical_dispersion_x, theoretical_math_e_x, dataset_len)

print('\nY intervals')
check_intervals(practical_dispersion_y, practical_math_e_y, theoretical_dispersion_y, theoretical_math_e_y, dataset_len)

print('\nCorellation')
check_corellation_interval(corellation_coeff_theoretical, corellation_coeff_practical, dataset_len)


###############################################################################
#                                  TASK B                                     #
###############################################################################
print('TASK B')

distribution = [
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

norm_sum = 0
for row in distribution: 
    for element in row:
        norm_sum += element
        
for j, row in enumerate(distribution):
    for i, element in enumerate(row):
        distribution[j][i] = element / norm_sum


def discrete_function(j=None, i=None):
    if i is None: 
        return sum(distribution[j])
    if j is None: 
        return sum([row[i] for row in distribution])
    return distribution[j][i]

class DRVGenerator:
    def __init__(self, seed=prog_seed, multiplier=uint16_t_MAX, divider=int32_t_MAX, dist=distribution):
        self.generator = CongruentGenerator(seed, multiplier, divider)
        self.dist = dist
    
    def generate_value(self):
        _x, _y = self.generator.generate_value(), self.generator.generate_value()
        
        i = 0
        accumulator = discrete_function(i=i)
        while _x > accumulator:
            i += 1
            accumulator += discrete_function(i=i)

        normalize = discrete_function(i=i)
        j = 0
        accumulator = discrete_function(i=i, j=j) / normalize
        while _y > accumulator:
            j += 1
            accumulator += discrete_function(i=i, j=j) / normalize
        
        return i, j

dataset_len = 10000

drv_generator = DRVGenerator()

dataset = [drv_generator.generate_value() for _ in range(dataset_len)]

zipped_data = list(zip(*dataset))

sns.jointplot(x=zipped_data[0],
              y=zipped_data[1], 
              kind="kde", 
              space=0)
plt.show()

dataset_len = 1000000
generator = DRVGenerator()
dataset = [generator.generate_value() for _ in range(dataset_len)]
zipped_data = list(zip(*dataset))

frequency = defaultdict(int)
for p in dataset:
    frequency[p] += (1 / dataset_len)

heat_map = [0 for _ in range(len(distribution))]
for i in range(len(distribution)):
    heat_map[i] = [0] * len(distribution[-1])

for x, y in sorted(frequency.keys()):
    heat_map[y][x] = frequency[(x, y)]

frameData = DataFrame(heat_map)
print('Empirical distribution matrix:\n', frameData)
sns.heatmap(data=frameData)
plt.show()

# to display density functions on hists
accumulated_vector_y = []

for i in range(len(distribution)):
    accumulated_vector_y.append(sum(distribution[i])) 

accumulated_vector_x = []

for j in range(len(distribution[-1])):
    accumulated_vector_x.append(sum([row[j] for row in distribution]))  

# display hists and density functions
sns.distplot(zipped_data[0], bins=11, kde=False, norm_hist=True)
x = np.linspace(0, 10, 11)
frameData = DataFrame(data={'x': x, 'y': accumulated_vector_x})
sns.lineplot(x='x', y='y', data=frameData, palette="ch:2.5,.25")
plt.show()

sns.distplot(zipped_data[1], bins=10, kde=False, norm_hist=True)
x = np.linspace(0, 9, 10)
frameData = DataFrame(data={'x': x, 'y': accumulated_vector_y})
sns.lineplot(x='x', y='y', data=frameData, palette="ch:2.5,.25")
plt.show()


# estimates
theoretical_math_e_x = 0
theoretical_math_e_y = 0
for i in range(len(distribution)):
    for j in range(len(distribution[i])):
        theoretical_math_e_x += distribution[i][j] * j
        theoretical_math_e_y += distribution[i][j] * i
        
x = zipped_data[0]
y = zipped_data[1]

practical_math_e_x = sum(x) / len(x)
practical_math_e_y = sum(y) / len(y)

print('Math exp:')
print('X:')
print('theoretical_math_e_x:', theoretical_math_e_x)
print('practical_math_e_x:', practical_math_e_x)
print('Y:')
print('theoretical_math_e_y:', theoretical_math_e_y)
print('practical_math_e_y:', practical_math_e_y)

theoretical_dispersion_x = 0
theoretical_dispersion_y = 0
for i in range(len(distribution)):
    for j in range(len(distribution[i])):
        theoretical_dispersion_x += distribution[i][j] * (j - theoretical_math_e_x) ** 2
        theoretical_dispersion_y += distribution[i][j] * (i - theoretical_math_e_y) ** 2

practical_dispersion_x = sum([(practical_math_e_x - item) ** 2 for item in x]) / (len(x) - 1)
practical_dispersion_y = sum([(practical_math_e_y - item) ** 2 for item in y]) / (len(y) - 1)

print('\nDispersion:')
print('X:')
print('theoretical_dispersion_x:', theoretical_dispersion_x)
print('practical_dispersion_x:', practical_dispersion_x)
print('Y:')
print('theoretical_dispersion_y:', theoretical_dispersion_y)
print('practical_dispersion_y:', practical_dispersion_y)

theoretical_math_e_xy = 0

for i in range(len(distribution)):
    for j in range(len(distribution[i])):
        theoretical_math_e_xy += distribution[i][j] * i * j
        
corellation_coeff_theoretical = (theoretical_math_e_xy - theoretical_math_e_x * theoretical_math_e_y) / np.sqrt(theoretical_dispersion_y * theoretical_dispersion_x)
corellation_coeff_practical = sum([(_x - practical_math_e_x) * (_y - practical_math_e_y) for _x, _y in dataset]) / dataset_len
corellation_coeff_practical /= (np.sqrt(practical_dispersion_x) * np.sqrt(practical_dispersion_y))

print('\nCorellation coeff:')
print('theoretical', corellation_coeff_theoretical)
print('practical', corellation_coeff_practical)

# moments check
print('\nX intervals')
check_intervals(practical_dispersion_x, practical_math_e_x, theoretical_dispersion_x, theoretical_math_e_x, dataset_len)

print('\nY intervals')
check_intervals(practical_dispersion_y, practical_math_e_y, theoretical_dispersion_y, theoretical_math_e_y, dataset_len)

print('\nCorellation')
check_corellation_interval(corellation_coeff_theoretical, corellation_coeff_practical, dataset_len)