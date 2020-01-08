import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.fftpack
import random


SEED = 4
COUNT = 10000
FREQUENCY = 2
DISPERSION = 2
ALPHA = 0.435
STEP = 0.83
WHITE_NOISE_MEAN = 0.0
WHITE_NOISE_DISPERSION = 1/3

np.random.seed(SEED)


def correlation_func(dispersion, alpha, tau):
    return np.exp(- alpha * np.abs(tau)) * (1 + alpha * np.abs(tau))


def white_noise_gen(dt=0, dispersion=WHITE_NOISE_DISPERSION):
    a = np.sqrt(3 * dispersion)
    return np.random.uniform(-a, a)


def plot_process_data(data, dt, label='process'):
    count = len(data)
    x = np.linspace(0, count * dt, count)
    plt.plot(x, data)
    plt.ylabel('{} plt:'.format(label))
    plt.show()


def build_hist(dataset):
    n = len(dataset)
    bins_count = 16
    n, _, _ = plt.hist(dataset, bins_count, density=True, facecolor='g', alpha=0.75)
    plt.title('Histogram data (normalized)')
    plt.xlim(min(dataset), max(dataset))
    plt.ylim(0, max(n))
    plt.grid(True)
    plt.show()


def build_corellation_func(mean, dataset):
    deviations = [item - mean for item in dataset]
    corellation_vals = np.correlate(deviations, deviations, mode='full')[-len(dataset):]
    corellation_vals /= max(corellation_vals)
    return corellation_vals


def autocorellation_psi_func(tau, i, dispersion=DISPERSION, alpha=ALPHA):
    return np.sqrt(dispersion) * alpha * tau * np.exp(-alpha * i * tau) * np.sqrt(2 / (alpha * np.pi))


def white_noise_phi_func(tau, i, dispersion=WHITE_NOISE_DISPERSION):
    return np.sqrt(dispersion)


def generate_output_signal(input_sig, psi_tau_func, phi_tau_func, tau, max_int=40):
    output_signal = []
    h_t_func = lambda tau, i: psi_tau_func(tau, i) / phi_tau_func(tau, i)
    for j, _ in enumerate(input_sig):
        y_j = 0
        for i in range(0, min(j + 1, max_int)):
            y_j += (tau * h_t_func(i, tau) * input_sig[j - i])
            
        output_signal.append(y_j)
    
    return output_signal


def check_intervals(non_offsetted_estimate_dispersion, math_estimate, real_dispersion, real_math_exp, dataset_len):
    gamma = 0.90
    math_exp_delta = (non_offsetted_estimate_dispersion ** 0.5) / (dataset_len - 1) ** 0.5 * scipy.stats.norm.ppf(gamma)
    
    math_lower_bound, math_upper_bound = real_math_exp - math_exp_delta, real_math_exp + math_exp_delta

    dispersion_lower_bound = dataset_len * non_offsetted_estimate_dispersion / scipy.stats.chi2.isf((1 - gamma) / 2, dataset_len - 1)
    dispersion_upper_bound = dataset_len * non_offsetted_estimate_dispersion / scipy.stats.chi2.isf((1 + gamma) / 2, dataset_len - 1)

    return math_lower_bound, math_upper_bound, dispersion_lower_bound, dispersion_upper_bound

    
def build_spectr(corr):
    spectr = scipy.fftpack.fft(corr)
    return np.abs(spectr)


def get_mean(dataset):
    return sum(dataset) / len(dataset)
    

def get_dispersion(dataset, alpha):
    mean = get_mean(dataset)
    return sum([(mean - item) ** 2 for item in dataset]) / (len(dataset) - 1)


def get_chi_zero_check(X_src):
    m = len(X_src)
    X = [x for x in X_src]
    
    def get_chi(X, n):
        chi = 0
        for i in range(len(X)):
            chi += (X[i]) ** 2 / abs(X[i])
            
        return chi * n
    
    chi = get_chi(X, m)
    chi_get = scipy.stats.chi2.isf(0.01, (m - 1))
    print('chi_current, chi_max:', chi, chi_get)
    return chi < chi_get


def get_chi_check(X_src, real_val):
    m = len(X_src)
    X = [x for x in X_src]
    
    def get_chi(real, X, n):
        chi = 0
        for i in range(len(X)):
            chi += (real - X[i]) ** 2 / (real * X[i])
            
        return chi * n
    
    real = real_val
    
    chi = get_chi(real, X, m)
    chi_get = scipy.stats.chi2.isf(0.01, (m - 1))
    print('chi_current, chi_max:', chi, chi_get)
    return chi < chi_get


def slutskey_check(corellation, tau, acceptance=0.01, T_max=10000):
    T = min(len(corellation), T_max)
    slutskey_value = 1 / T * sum([value * (1 - i * tau / T) for i, value in enumerate(corellation[:T])])
    if slutskey_value < acceptance:
        print('slutskey accepted! Process is ergodicit! Value - {}'.format(slutskey_value))
    else:
        print('slutskey denied! Process is not ergodicit! Value - {}'.format(slutskey_value))


def dicky_fuller(noise):
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(noise)
      
    return result[4]['10%'] < result[1]
    

def stationarity_check(output_signal, true_mean, alpha, dispersion=DISPERSION, batches=30):
    print('Stationary check:')
    mean_list = []
    dispersion_list = []
    batch_size = 4 * len(output_signal) // batches
    
    for _ in range(batches):
        batch = random.sample(output_signal, batch_size)
        mean_list.append(get_mean(batch))
        dispersion_list.append(get_dispersion(batch, alpha))
    
    return get_chi_check(dispersion_list, dispersion), get_chi_zero_check(mean_list)

   

class DataGenerator:
    def __init__(self, func):
        self.func = func
        
    def generate_noise(self, n, dt):
        return np.array([self.func(dt) for i in range(n)])


class BrownianDataGenerator:
    def __init__(self, sub_func):
        self.sub_func = sub_func
        
    def generate_noise(self, n, dt):
        max_int = 40
        sub_vals = np.array([self.sub_func(dt) for i in range(n)])
        
        result = []
        for j, _ in enumerate(sub_vals):
            y_j = 0
            for i in range(0, min(j + 1, max_int)):
                y_j += (dt * sub_vals[j - i])
                
            result.append(y_j)
        
        return result


if __name__ == '__main__':
    data_generator = DataGenerator(white_noise_gen)

    white_noise_dataset = data_generator.generate_noise(COUNT, STEP)
    plot_process_data(white_noise_dataset, STEP, 'white noise')
    build_hist(white_noise_dataset)

    print('White noise mean', get_mean(white_noise_dataset))

    white_noise_output = generate_output_signal(white_noise_dataset, autocorellation_psi_func, white_noise_phi_func, STEP)
    plot_process_data(white_noise_output, STEP, 'result process')
    build_hist(white_noise_output)

    practical_math_e = get_mean(white_noise_output)
    print(f'Math exp) {WHITE_NOISE_MEAN} ~ {practical_math_e}')

    practical_dispersion = get_dispersion(white_noise_output, ALPHA)
    print(f'Dispersion) {DISPERSION} ~ {practical_dispersion}')

    m1, m2, d1, d2 = check_intervals(practical_dispersion, practical_math_e, DISPERSION, WHITE_NOISE_MEAN, len(white_noise_output))
    print(f'Math exp: {m1} <= {practical_math_e} <= {m2}')
    print(f'Math exp: {d1} <= {practical_dispersion} <= {d2}')

    corellation_theory = []
    i = 0
    while True:
        corellation_theory.append(correlation_func(DISPERSION, ALPHA, STEP * i))
        i += 1
        if corellation_theory[-1] < 0.05:
            break

    corellation_timeline_count = i

    corellation_practical = build_corellation_func(practical_math_e, white_noise_output)[:corellation_timeline_count]

    count = len(corellation_theory)
    x = np.linspace(0, count * STEP, count)
    plt1 = plt.plot(x, corellation_theory)
    plt.ylabel('{} plt:'.format('corellations'))

    count = len(corellation_practical)
    x = np.linspace(0, count * STEP, count)
    plt2 = plt.plot(x, corellation_practical)

    plt.legend((plt1[0], plt2[0]), ('theoretical', 'practical'))
        
    plt.show()

    spectr_practical = build_spectr(corellation_practical)
    x = np.linspace(0, count * STEP, count)
    plt1 = plt.plot(x, spectr_practical)
    plt.ylabel('{} plt:'.format('specters'))

    spectr_theory = build_spectr(corellation_theory)
    x = np.linspace(0, count * STEP, count)
    plt2 = plt.plot(x, spectr_theory)

    plt.legend((plt1[0], plt2[0]), ('practical', 'theoretical' ))
        
    plt.show()

    corellation_practical = build_corellation_func(practical_math_e, white_noise_output)
    slutskey_check(corellation_practical, STEP)


    is_pass_despersion, is_pass_mean = stationarity_check(white_noise_output, WHITE_NOISE_MEAN, ALPHA)

    print('Chi test passed for dispersion!' if is_pass_despersion else 'Chi test rejected for dispersion!')
    print('Chi test passed for mean!' if is_pass_mean else 'Chi test rejected for mean!')
    print('Process is stational!' if dicky_fuller(white_noise_output) else 'Process is non stational!')