import numpy as np
import scipy as sp
import math
from numpy.typing import ArrayLike
from typing import Callable

augmentation_registry: dict[str, Callable[[ArrayLike, np.ndarray, float, list, dict], float]] = {}

def register_augmentation(name: str) -> Callable:
    """
    Decorator to register an augmentation method.

    Args:
        name (str): Name of the augmentation method to register.

    Returns:
        Callable: The decorator that registers the augmentation function.
    """
    def wrapper(func: Callable[[ArrayLike, np.ndarray, float, list, dict], float]) -> Callable:
        augmentation_registry[name] = func
        return func
    return wrapper

def get_augmentation(name: str) -> Callable[[ArrayLike, np.ndarray, float, list, dict], float]:
    """
    Retrieves the augmentation function by name.

    Args:
        name (str): The name of the augmentation method.

    Returns:
        Callable: The registered augmentation function.
    """
    return augmentation_registry[name]


@register_augmentation("linear")
def linear_augmentation(
    u: ArrayLike,
    A_row: np.ndarray,
    infection_time: float,
    path: list,
    edges: dict
) -> float:
    """
    Linear approximation for conditional joint MGF.

    Args:
        u (ArrayLike): Input vector.
        A_row (np.ndarray): A-matrix row for source.
        infection_time (float): Infection time for the first observer.
        path (list): List of edges on the infection path.
        edges (dict): Dictionary of edge objects.

    Returns:
        float: The linear approximation factor.
    """
    approx_value = 0.0
    for i, edge in enumerate(edges.keys()):
        approx_value += np.matmul(u, A_row[:, i])
    approx_value *= -infection_time / len(path)
    return np.exp(approx_value)


@register_augmentation("exponential")
def exponential_augmentation(
    u: ArrayLike,
    A_row: np.ndarray,
    infection_time: float,
    path: list,
    edges: dict
) -> float:
    """
    Exponential approximation for conditional joint MGF.

    Args:
        u (ArrayLike): Input vector.
        A_row (np.ndarray): A-matrix row for source.
        infection_time (float): Infection time for the first observer.
        path (list): List of edges on the infection path.
        edges (dict): Dictionary of edge objects.

    Returns:
        float: The exponential approximation factor.
    """
    b1 = 0.0
    b2 = 0.0
    for i, edge in enumerate(edges.keys()):
        b2 += edges[edge].mgf_derivative2(0) - edges[edge].mgf_derivative(0)**2
        b1 += np.matmul(u, A_row[:, i]) * b2
    b = b1 / b2
    a1 = 0.0
    for i, edge in enumerate(edges.keys()):
        a1 += (b - np.matmul(u, A_row[:, i])) * edges[edge].mgf_derivative(0)
    a = np.exp(a1)
    return a * np.exp(-b * infection_time)


@register_augmentation("exact")
def exact_exponential_augmentation(
    u: ArrayLike,
    A_row: np.ndarray,
    infection_time: float,
    path: list,
    edges: dict
) -> float:
    """
    Exact exponential solution for iid exponential delays.

    Args:
        u (ArrayLike): Input vector.
        A_row (np.ndarray): A-matrix row for source.
        infection_time (float): Infection time for the first observer.
        path (list): List of edges on the infection path.
        edges (dict): Dictionary of edge objects.

    Returns:
        float: The exact exponential conditional MGF factor.
    """
    Theta = np.zeros((len(path), len(path)))
    lam = -1
    prod = 1.0
    for i, edge in enumerate(path):
        if i == 0:
            lam = edges[edge].params['lambda']
        prod *= 1 / (lam + np.matmul(u, A_row[:, i]))
        Theta[i, i] = -1 * (lam + np.matmul(u, A_row[:, i]))
        if i != len(path) - 1:
            Theta[i, i + 1] = lam + np.matmul(u, A_row[:, i])
    alpha = np.zeros((1, len(path)))
    alpha[0, 0] = 1
    exp_Theta = sp.linalg.expm(infection_time * Theta)
    g_t = -1 * np.matmul(np.matmul(np.matmul(alpha, exp_Theta), Theta), np.ones((1, len(path), 1)))
    return g_t * (infection_time ** (len(path) - 1)) * np.exp(-lam * infection_time) * math.factorial(len(path) - 1) * prod