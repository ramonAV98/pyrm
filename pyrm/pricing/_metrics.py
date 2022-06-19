from abc import ABCMeta, abstractmethod


def make_pricing_metric(name, d, **kwargs):
    """Factory function for crating pricing metric instances.

    Parameters
    ----------
    name : str
        Pricing metric name.

    d : callable
        Price response function. Must be:
            - nonnegative
            - continuous
            - differentiable

    kwargs : Key-word args.
        Additional arguments for pricing metrics.

    Returns
    -------
    object : PricingMetric object
    """
    metrics = {
        'revenue': Revenue,
        'total_contribution': TotalContribution}
    return metrics[name](d, **kwargs)


class PricingMetric(metaclass=ABCMeta):
    """A generic pricing metric class meant for subclassing.

    `PricingMetric` is a base class to construct specific pricing metrics
    classes and instances. It cannot be used directly.
    """

    def __init__(self, d, **kwargs):
        self.d = d
        vars(self).update(kwargs)

    @abstractmethod
    def f(self, p):
        """The actual metric function.

        All derived classes must implement their own metric/objective function.
        """
        pass


class Revenue(PricingMetric):
    def __init__(self, d):
        super().__init__(d)

    def f(self, p):
        return p * self.d(p)


class TotalContribution(PricingMetric):
    def __init__(self, d, c):
        super().__init__(d, c=c)

    def f(self, p):
        return (p - self.c) * self.d(p)
