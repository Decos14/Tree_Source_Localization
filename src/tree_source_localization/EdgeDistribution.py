from abc import ABC, abstractmethod
import numpy as np
from tree_source_localization import MGF_Functions
import scipy as sp

class BaseDistribution(ABC):
    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def sample(self) -> float: ...
    @abstractmethod
    def mgf(self, t: float) -> float: ...
    @abstractmethod
    def mgf_derivative(self, t: float) -> float: ...
    @abstractmethod
    def mgf_derivative2(self, t: float) -> float: ...

class DistributionRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(dist_class):
            cls._registry[name] = dist_class
            return dist_class
        return wrapper

    @classmethod
    def create(cls, name: str, params: dict) -> BaseDistribution:
        if name not in cls._registry:
            raise ValueError(f"Unknown distribution type: {name}")
        return cls._registry[name](params)

@DistributionRegistry.register("N")
class PositiveNormalDistribution(BaseDistribution):
    def __init__(self, params):
        super().__init__(params)
        self.mu = params['mu']
        self.sigma2 = params['sigma2']
        self.type = "N"

    def sample(self) -> float:
        return np.random.normal(self.mu, self.sigma2)

    def mgf(self, t: float) -> float:
        return 1 if np.isclose(t, 0) else MGF_Functions.PositiveNormalMGF(t, self.mu, self.sigma2)

    def mgf_derivative(self, t: float) -> float:
        return MGF_Functions.PositiveNormalMGFDerivative(t, self.mu, self.sigma2)

    def mgf_derivative2(self, t: float) -> float:
        return MGF_Functions.PositiveNormalMGFDerivative2(t, self.mu, self.sigma2)

@DistributionRegistry.register("E")
class ExponentialDistribution(BaseDistribution):
    def __init__(self, params):
        super().__init__(params)
        self.lam = params['lambda']
        self.type = "E"

    def sample(self) -> float:
        return np.random.exponential(self.lam)

    def mgf(self, t: float) -> float:
        return 1 if np.isclose(t, 0) else MGF_Functions.ExponentialMGF(t, self.lam)

    def mgf_derivative(self, t: float) -> float:
        return MGF_Functions.ExponentialMGFDerivative(t, self.lam)

    def mgf_derivative2(self, t: float) -> float:
        return MGF_Functions.ExponentialMGFDerivative2(t, self.lam)

@DistributionRegistry.register("U")
class UniformDistribution(BaseDistribution):
    def __init__(self, params):
        super().__init__(params)
        self.start = params['start']
        self.stop = params['stop']
        self.type = "U"

    def sample(self) -> float:
        return np.random.uniform(self.start, self.stop)

    def mgf(self, t: float) -> float:
        return MGF_Functions.UniformMGF(t, self.start, self.stop)

    def mgf_derivative(self, t: float) -> float:
        return MGF_Functions.UniformMGFDerivative(t, self.start, self.stop)

    def mgf_derivative2(self, t: float) -> float:
        return MGF_Functions.UniformMGFDerivative2(t, self.start, self.stop)

@DistributionRegistry.register("P")
class PoissonDistribution(BaseDistribution):
    def __init__(self, params):
        super().__init__(params)
        self.lam = params['lambda']
        self.type = "P"

    def sample(self) -> float:
        return np.random.poisson(self.lam)

    def mgf(self, t: float) -> float:
        return MGF_Functions.PoissonMGF(t, self.lam)

    def mgf_derivative(self, t: float) -> float:
        return MGF_Functions.PoissonMGFDerivative(t, self.lam)

    def mgf_derivative2(self, t: float) -> float:
        return MGF_Functions.PoissonMGFDerivative2(t, self.lam)

@DistributionRegistry.register("C")
class AbsoluteCauchyDistribution(BaseDistribution):
    def __init__(self, params):
        super().__init__(params)
        self.sigma2 = params['sigma2']
        self.type = "C"

    def sample(self) -> float:
        return np.abs(sp.stats.cauchy.rvs(loc=0,scale = self.sigma2))

    def mgf(self, t: float) -> float:
        return 1 if np.isclose(t, 0) else MGF_Functions.AbsoluteCauchyMGF(t, self.sigma2)

    def mgf_derivative(self, t: float) -> float:
        return MGF_Functions.AbsoluteCauchyMGFDerivative(t, self.sigma2)

    def mgf_derivative2(self, t: float) -> float:
        return MGF_Functions.AbsoluteCauchyMGFDerivative2(t, self.sigma2)

class EdgeDistribution:
    def __init__(self, dist_type: str, params: dict):
        self.dist_type = dist_type
        self.params = params
        self.delay = 0
        self.impl = DistributionRegistry.create(dist_type, params)

    def sample(self) -> float:
        self.delay = self.impl.sample()

    def mgf(self, t: float) -> float:
        return self.impl.mgf(t)

    def mgf_derivative(self, t: float) -> float:
        return self.impl.mgf_derivative(t)

    def mgf_derivative2(self, t: float) -> float:
        return self.impl.mgf_derivative2(t)