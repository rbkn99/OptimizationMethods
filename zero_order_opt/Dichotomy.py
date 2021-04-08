import BaseOpt


class Dichotomy(BaseOpt):
    def _step(self, x1, x2):
        margin = self.eps / 3
        m = (x1 + x2) / 2
        return m - margin, m + margin
