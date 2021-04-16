from sympy import Derivative, symbols, hessian
import numpy as np


class FunctionWrapper:
    def __init__(self, f, variables=('x', 'y')):
        self.f = f
        self.vars = symbols(' '.join(variables))
        self.invocations = 0
        self.grad_invocations_counter = 0
        self.hessian_invocations_counter = 0
        self._grad = [Derivative(f, var).doit() for var in self.vars]
        self._hessian = hessian(f, self.vars)
        self.require_count = False

    def __call__(self, *args, **kwargs):
        if self.require_count:
            self.invocations = self.invocations + 1
        return self.f(*args, **kwargs)

    def grad(self, xs):
        if self.require_count:
            self.grad_invocations_counter += 1
        return np.array([gradient.subs(list(zip(self.vars, xs))) for gradient in self._grad])

    def hessian(self, xs):
        if self.require_count:
            self.hessian_invocations_counter += 1
        return np.array([hs.subs(list(zip(self.vars, xs))) for hs in self._hessian])

    def stop_count(self):
        self.require_count = False

    def start_count(self):
        self.require_count = True

    def reset(self):
        self.require_count = False
        self.invocations = 0
        self.grad_invocations_counter = 0
        self.hessian_invocations_counter = 0
