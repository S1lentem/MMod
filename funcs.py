import numpy as np

def get_intervals(a, b, count, values):
    intervals = [[i, []] for i in np.arange(a, b, (b-a)/count)] + [[float(b), []]]
    
    for value in values:
        for i in range(len(intervals) - 1):
            if value >= intervals[i][0] and value < intervals[i+1][0]:
                intervals[i][-1].append(value)
                break
    
    return intervals


def chi_2(interval, theory_func, item_count):
    result = 0
    for i, item in enumerate(interval[:-1:]):
        o = len(item[-1]) / item_count
        
        temp = item[0] + interval[i+1][0] / 2
        e = theory_func(temp)

        result += (o - e)**2 / e

    return result