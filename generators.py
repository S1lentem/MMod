RANDOM_VALUE = 0
PROBALITY = 1

class BaseRandomGeneretor:
    SEED = 1
    K = 16807
    M = 0x7fffffff


    def __init__(self):
        self.prev = self.SEED


    def next(self):
        self.prev = (self.K*self.prev) % self.M 
        return self.prev / self.M


    def reset(self):
        self.prev = self.SEED


    def get_iterator(self, iteration_count):
        for _ in range(iteration_count):
            yield self.next()




class ContinuousRandomNumberGenerator:
    def __init__(self, func, max_y):
        self.__generator = BaseRandomGeneretor()
        self.__func = func
        self.__max_y = max_y


    def reset(self):
        self.__generator.reset()


    def next(self, b_x, b_y, a_x=0, a_y=0):
        while True:
            x_value = a_x + (b_x - a_x) * self.__generator.next()
            # y_value = a_y + (b_y - a_y) * self.__generator.next()
            y_value = self.__max_y * self.__generator.next()
            if self.__func(x_value) > y_value:
                return x_value


    def func(self, x):
        return self.__func(x)


    def get_iterator(self, count_iteration, b_x, b_y, a_x=0, a_y=0):
        for _ in range(count_iteration):
            yield self.next(b_x, b_y, a_x, a_y)



class DiscreteRandomGenerator2:
    def __init__(self, distribution):
        self.__generator = BaseRandomGeneretor()
        self.__distribution = distribution
        self.__intervals = [0]

        for value in distribution:
            self.__intervals.append(self.__intervals[-1] + value[-1])


    def __get_discrete_value(self, value):
        for i in range(len(self.__intervals) - 1):
            if value >= self.__intervals[i] and value < self.__intervals[i+1]:
                return self.__distribution[i][0]


    def reset(self):
        self.__generator.reset()


    def next(self):
        value = self.__generator.next()
        return self.__get_discrete_value(value)


    def get_iterator(self, count):
        for _ in range(count):
            yield self.next()



class DiscreteRandomGenerator:
    def __init__(self, interval, func, **args):
        self.__generator = BaseRandomGeneretor()
        self.__probability_range = []
        self.__intervals = [0]

        for i in interval:
            probality = func(i=i, **args)
            self.__probability_range.append((i, probality))
            self.__intervals.append(self.__intervals[-1] + probality)


    def reset(self):
        self.__generator.reset()


    def next(self):
        value = self.__generator.next()
        return self.__get_discrete_value(value)


    def get_iterator(self, count):
        for _ in range(count):
            yield self.next()
     
    def get_probality_range(self):
        return self.__probability_range

    def __get_discrete_value(self, base_value):
        for i in range(len(self.__intervals) - 1):
            if base_value >= self.__intervals[i] and base_value < self.__intervals[i+1]:
                return self.__probability_range[i][RANDOM_VALUE]
        
        return self.__probability_range[-1][RANDOM_VALUE]
