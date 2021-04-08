from abc import abstractmethod


class BaseOpt:
    def __init__(self, f, eps, left_bound, right_bound):
        self.f = f
        self.eps = eps
        self.left_bound = left_bound
        self.right_bound = right_bound

    def optimize(self):
        left = self.left_bound
        right = self.right_bound
        while right - left > self.eps:
            x1, x2 = self._step(left, right)
            if self.f(x1) < self.f(x2):
                right = x2
            else:
                left = x1
        return (left + right) / 2

    @abstractmethod
    def _step(self, x1, x2):
        pass
