import BaseOpt
import math


class GoldenRatio(BaseOpt):
    def __init__(self, f, eps, left_bound, right_bound):
        super().__init__(self, f, eps, left_bound, right_bound)
        self.golden_ratio = (math.sqrt(5) - 1) / 2

    def _step(self, x1, x2):
        diff = x2 - x1
        return x2 - self.golden_ratio * diff, x1 + self.golden_ratio * diff
