from abc import abstractmethod, ABC
import numpy as np
import pandas as pd
import math


class GoldenRatio(ABC):
    def __init__(self, f, eps, bounds):
        self.eps = eps
        self.f = f
        self.left = bounds[0]
        self.right = bounds[1]
        self._log = []
        self._log_headers = ['Iter', 'l', 'r']
        self._log_headers = self._log_headers + ['x1', 'x2']
        self.golden_c = (math.sqrt(5) - 1) / 2

    def _opt_inner(self):
        left = self.left
        right = self.right

        x1, x2 = self.make_step(left, right)
        self._log.append([left, right, x1, x2])
        f1 = self.f(x1)
        f2 = self.f(x2)
        while True:
            if f1 < f2:
                right = x2
                if right - left < self.eps:
                    break
                x1, x2 = self.make_step(left, right)
                f2 = f1
                f1 = self.f(x1)
            else:
                left = x1
                if right - left < self.eps:
                    break
                x1, x2 = self.make_step(left, right)
                f1 = f2
                f2 = self.f(x2)

            self._log.append([left, right, x1, x2])

        return (left + right) / 2

    def make_step(self, x1, x2):
        diff = x2 - x1
        return x2 - self.golden_c * diff, x1 + self.golden_c * diff


class BaseOptimizer:
    def __init__(self, f, eps, preserve_logs=True):
        self.eps = eps
        self.f = f
        self._log = []
        self.base_optimizer = GoldenRatio
        self.preserve_logs = preserve_logs

        self.DIVERGE_BORDER = 1e9
        self.MAX_ITERATIONS = int(1e5)

    def get_n_iterations(self):
        return len(self._log) - 1

    @abstractmethod
    def make_step(self, x):
        pass

    def optimize(self, init_points):
        cur_state = np.array(init_points)
        self.f.reset()
        if self.preserve_logs:
            self._log = [list(init_points) + [self.f(*init_points)]]
        else:
            self._log = []
        self.f.start_count()
        for i in range(self.MAX_ITERATIONS):
            new_state = self.make_step(cur_state)
            diff = new_state - cur_state
            if np.linalg.norm(diff) > self.DIVERGE_BORDER:
                raise RuntimeError("Optimizer diverges")
            if np.linalg.norm(diff) < self.eps:
                self.f.stop_count()
                return cur_state
            cur_state = new_state
            if self.preserve_logs:
                self.f.stop_count()
                self._log.append(list(cur_state) + [self.f(*cur_state)])
                self.f.start_count()
        raise RuntimeError("Reach max steps counter")

    def log_frame(self):
        if not self.preserve_logs:
            raise RuntimeError("Log not preserved")
        return pd.DataFrame([[i] + self._log[i] for i in range(len(self._log))],
                            columns=['index', 'x', 'y', 'f'])
