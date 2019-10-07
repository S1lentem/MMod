def projection(begin, end, M):
    def decorate(func):
        def projection_decorate(*args, **kwargs):
            result = func(*args, **kwargs)
            return (result * (end - begin) / M) + begin
        return projection_decorate
    return decorate


class BaseRandomGeneretor:
    SEED = 1
    K = 16807
    M = 0x7fffffff

    def __init__(self):
        self.prev = self.SEED
    
    @projection(0, 1, M)
    def next(self):
        self.prev = (self.K*self.prev) % self.M 
        return self.prev

    def reset(self):
        self.prev = self.SEED
