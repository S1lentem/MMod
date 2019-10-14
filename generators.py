


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
    def __init__(self, func):
        self.__generator = BaseRandomGeneretor()
        self.__func = func
        
    def reset(self):
        self.__generator.reset()

    def next(self, b_x, b_y, a_x=0, a_y=0):
        x_value = a_x + (b_x - a_x) * self.__generator.next()
        y_value = a_y + (b_y - a_y) * self.__generator.next()
        return x_value if self.__func(x_value) > y_value else self.next(b_x, b_y, a_x, a_y)

    def func(self, x):
        return self.__func(x)
    
    def get_iterator(self, count_iteration, b_x, b_y, a_x=0, a_y=0):
        for _ in range(count_iteration):
            yield self.next(b_x, b_y, a_x, a_y)