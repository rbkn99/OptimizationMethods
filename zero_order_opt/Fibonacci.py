import BaseOpt


class Fibonacci(BaseOpt):
    def __init__(self, f, eps, left_bound, right_bound):
        super().__init__(self, f, eps, left_bound, right_bound)
        self.fib_seq = []

    def __calc_fib_seq(self):
        self.fib_seq = [1, 1, 2]
        while (self.right_bound - self.left_bound) / self.eps * 2 >= self.fib_seq[-1]:
            self.fib_seq.append(self.fib_seq[-1] + self.fib_seq[-2])
