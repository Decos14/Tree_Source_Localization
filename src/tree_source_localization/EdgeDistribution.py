from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Dict, Type

import numpy as np
import scipy as sp

from tree_source_localization import MGFFunctions


class BaseDistribution(ABC):
    def __init__(self, params: Dict[str, float]) -> None:
        for key, value in params.items():
            if not isinstance(key, str) or not isinstance(value, float):
                raise ValueError("Parameters are not of the correct format: Dict[str, float]")
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
    _registry: ClassVar[Dict[str, Callable]] = {}

    def __init__(self) -> None:
        pass

    @classmethod
    def register(cls, name: str) -> Callable[[Type[BaseDistribution]], Type[BaseDistribution]]:
        def wrapper(dist_class: Type[BaseDistribution]) -> Type[BaseDistribution]:
            cls._registry[name] = dist_class
            return dist_class

        return wrapper

    @classmethod
    def create(cls, name: str, params: Dict[str, float]) -> BaseDistribution:
        if name not in cls._registry:
            raise ValueError(f"Unknown distribution type: {name}")
        return cls._registry[name](params)


@DistributionRegistry.register("N")
class PositiveNormalDistribution(BaseDistribution):
    def __init__(self, params: Dict[str, float]) -> None:
        super().__init__(params)
        if "mu" not in params.keys() or "sigma2" not in params.keys():
            raise ValueError(
                f"Incorrect Parameters for positive normal distribution: {list(params.keys())}, "
                f"should be 'mu' and 'sigma2'"
            )
        self.mu = params["mu"]
        self.sigma2 = params["sigma2"]
        self.type = "N"

    def sample(self) -> float:
        val = -1
        while val <= 0:
            val = np.random.normal(self.mu, self.sigma2)
        return val

    def mgf(self, t: float) -> float:
        return 1 if np.isclose(t, 0) else MGFFunctions.positive_normal_mgf(t, self.mu, self.sigma2)

    def mgf_derivative(self, t: float) -> float:
        return MGFFunctions.positive_normal_mgf_derivative(t, self.mu, self.sigma2)

    def mgf_derivative2(self, t: float) -> float:
        return MGFFunctions.positive_normal_mgf_derivative2(t, self.mu, self.sigma2)


@DistributionRegistry.register("E")
class ExponentialDistribution(BaseDistribution):
    def __init__(self, params: Dict[str, float]) -> None:
        super().__init__(params)
        if "lambda" not in params.keys():
            raise ValueError("Incorrect Parameter for exponential distribution, should be 'lambda")
        self.lam = params["lambda"]
        self.type = "E"

    def sample(self) -> float:
        return np.random.exponential(self.lam)

    def mgf(self, t: float) -> float:
        return 1 if np.isclose(t, 0) else MGFFunctions.exponential_mgf(t, self.lam)

    def mgf_derivative(self, t: float) -> float:
        return MGFFunctions.exponential_mgf_derivative(t, self.lam)

    def mgf_derivative2(self, t: float) -> float:
        return MGFFunctions.exponential_mgf_derivative2(t, self.lam)


@DistributionRegistry.register("U")
class UniformDistribution(BaseDistribution):
    def __init__(self, params: Dict[str, float]) -> None:
        super().__init__(params)
        if "start" not in params.keys() and "stop" not in params.keys():
            raise ValueError("Incorrect parameters for uniform distribution, should be 'start' and 'stop'")
        self.start = params["start"]
        self.stop = params["stop"]
        self.type = "U"

    def sample(self) -> float:
        return np.random.uniform(self.start, self.stop)

    def mgf(self, t: float) -> float:
        return MGFFunctions.uniform_mgf(t, self.start, self.stop)

    def mgf_derivative(self, t: float) -> float:
        return MGFFunctions.uniform_mgf_derivative(t, self.start, self.stop)

    def mgf_derivative2(self, t: float) -> float:
        return MGFFunctions.uniform_mgf_derivative2(t, self.start, self.stop)


@DistributionRegistry.register("P")
class PoissonDistribution(BaseDistribution):
    def __init__(self, params: Dict[str, float]) -> None:
        super().__init__(params)
        if "lambda" not in params.keys():
            raise ValueError("Incorrect Parameter for poisson distribution, should be 'lambda")
        self.lam = params["lambda"]
        self.type = "P"

    def sample(self) -> float:
        return np.random.poisson(self.lam)

    def mgf(self, t: float) -> float:
        return MGFFunctions.poisson_mgf(t, self.lam)

    def mgf_derivative(self, t: float) -> float:
        return MGFFunctions.poisson_mgf_derivative(t, self.lam)

    def mgf_derivative2(self, t: float) -> float:
        return MGFFunctions.poisson_mgf_derivative2(t, self.lam)


@DistributionRegistry.register("C")
class AbsoluteCauchyDistribution(BaseDistribution):
    def __init__(self, params: Dict[str, float]) -> None:
        super().__init__(params)
        if "sigma2" not in params.keys():
            raise ValueError("Incorrect Parameters for absolute cauchy distribution, should be 'sigma2'")
        self.sigma2 = params["sigma2"]
        self.type = "C"

    def sample(self) -> float:
        return np.abs(sp.stats.cauchy.rvs(loc=0, scale=self.sigma2))

    def mgf(self, t: float) -> float:
        return 1 if np.isclose(t, 0) else MGFFunctions.absolute_cauchy_mgf(t, self.sigma2)

    def mgf_derivative(self, t: float) -> float:
        return MGFFunctions.absolute_cauchy_mgf_derivative(t, self.sigma2)

    def mgf_derivative2(self, t: float) -> float:
        return MGFFunctions.absolute_cauchy_mgf_derivative2(t, self.sigma2)


class EdgeDistribution:
    def __init__(self, dist_type: str, params: Dict[str, float]) -> None:
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
