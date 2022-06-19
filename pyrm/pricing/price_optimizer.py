from scipy.optimize import minimize_scalar


class PriceOptimizer:
    """
    """
    def __init__(self, metric):
        self.metric = metric

    def optimize(self):
        def fun(p):
            return - self.metric.f(p)

        res = minimize_scalar(fun)
        return res
