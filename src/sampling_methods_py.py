"""
Sampling Methods Module

This module implements various sampling strategies for active inference:
    1. Uniform Poisson Sampling: Equal inclusion probabilities
    2. Active Poisson Sampling: Uncertainty-weighted probabilities
    3. Cube Active Sampling: Balanced sampling with auxiliary variables
    4. Classical Simple Random Sampling: Traditional baseline

Each method returns sampling indicators and associated statistics.
"""

import numpy as np
from typing import Tuple, Optional
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

# Activate automatic conversion between NumPy and R
numpy2ri.activate()

# Import R's BalancedSampling package
try:
    BalancedSampling = importr('BalancedSampling')
    cube = BalancedSampling.cube
except Exception as e:
    print(f"Warning: Could not import BalancedSampling package: {e}")
    print("Cube sampling will not be available.")
    cube = None


def uniform_poisson_sampling(
    N: int,
    budget: float,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform uniform Poisson sampling with constant inclusion probability.
    
    Each unit is independently selected with probability equal to the
    sampling budget.
    
    Args:
        N: Population size
        budget: Sampling budget (target sampling rate)
        random_state: Random seed for reproducibility (default: None)
    
    Returns:
        sampling_indicators: Binary array (1 if sampled) of shape (N,)
        inclusion_probs: Array of inclusion probabilities of shape (N,)
    
    Example:
        >>> xi, pi = uniform_poisson_sampling(N=1000, budget=0.1)
        >>> print(f"Sampled {xi.sum()} out of {N} units")
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Constant inclusion probability for all units
    inclusion_probs = np.full(N, budget)
    
    # Independent Bernoulli sampling
    sampling_indicators = np.random.binomial(1, inclusion_probs)
    
    return sampling_indicators, inclusion_probs


def active_poisson_sampling(
    uncertainty: np.ndarray,
    budget: float,
    tau: float = 0.5,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform active Poisson sampling with uncertainty-weighted probabilities.
    
    Inclusion probabilities are computed as a convex combination of:
        - Uncertainty-proportional allocation: τ * (uncertainty / mean_uncertainty) * budget
        - Uniform allocation: (1 - τ) * budget
    
    The parameter τ controls the balance between active and uniform sampling.
    
    Args:
        uncertainty: Uncertainty estimates of shape (N,)
        budget: Sampling budget (target sampling rate)
        tau: Weight for uncertainty-based allocation, in [0, 1] (default: 0.5)
        random_state: Random seed for reproducibility (default: None)
    
    Returns:
        sampling_indicators: Binary array (1 if sampled) of shape (N,)
        inclusion_probs: Array of inclusion probabilities of shape (N,)
    
    Example:
        >>> xi, pi = active_poisson_sampling(uncertainty, budget=0.1, tau=0.5)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    N = len(uncertainty)
    mean_uncertainty = uncertainty.mean()
    
    # Compute inclusion probabilities as convex combination
    # π = τ * (u / ū) * b + (1 - τ) * b
    inclusion_probs = (
        tau * (uncertainty / mean_uncertainty) * budget +
        (1.0 - tau) * budget
    )
    
    # Ensure probabilities are in [0, 1]
    inclusion_probs = np.minimum(inclusion_probs, 1.0)
    
    # Independent Bernoulli sampling
    sampling_indicators = np.random.binomial(1, inclusion_probs)
    
    return sampling_indicators, inclusion_probs


def cube_active_sampling(
    auxiliary_vars: np.ndarray,
    inclusion_probs: np.ndarray,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform cube (balanced) sampling with auxiliary variables.
    
    Cube sampling selects a sample that is approximately balanced on
    the provided auxiliary variables while respecting the inclusion
    probabilities.
    
    This implementation uses the R package 'BalancedSampling' through rpy2.
    
    Args:
        auxiliary_vars: Auxiliary variables of shape (N, p)
        inclusion_probs: Inclusion probabilities of shape (N,)
        random_state: Random seed for reproducibility (default: None)
    
    Returns:
        sampling_indicators: Binary array (1 if sampled) of shape (N,)
        inclusion_probs: Array of inclusion probabilities of shape (N,)
    
    Raises:
        RuntimeError: If BalancedSampling package is not available
    
    Reference:
        Deville, J.-C., & Tillé, Y. (2004). Efficient balanced sampling.
        Biometrika, 91(4), 893-912.
    
    Example:
        >>> xi, pi = cube_active_sampling(uncertainty.reshape(-1, 1), pi)
    """
    if cube is None:
        raise RuntimeError(
            "BalancedSampling package not available. "
            "Please install it in R: install.packages('BalancedSampling')"
        )
    
    if random_state is not None:
        np.random.seed(random_state)
    
    N = len(inclusion_probs)
    
    # Ensure auxiliary_vars is 2D
    if auxiliary_vars.ndim == 1:
        auxiliary_vars = auxiliary_vars.reshape(-1, 1)
    
    # Convert to R objects
    auxiliary_vars_r = numpy2ri.py2rpy(auxiliary_vars)
    inclusion_probs_r = numpy2ri.py2rpy(inclusion_probs)
    
    # Call cube sampling from R
    cube_result = cube(auxiliary_vars_r, prob=inclusion_probs_r)
    
    # Convert selected indices back to Python (R uses 1-based indexing)
    selected_indices = np.array(cube_result).astype(int) - 1
    
    # Create sampling indicators
    sampling_indicators = np.zeros(N, dtype=int)
    sampling_indicators[selected_indices] = 1
    
    return sampling_indicators, inclusion_probs


def classical_simple_random_sampling(
    N: int,
    budget: float,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform classical simple random sampling (SRS).
    
    This is a baseline method that does not use predictions. Each unit
    is independently selected with equal probability.
    
    Args:
        N: Population size
        budget: Sampling budget (target sampling rate)
        random_state: Random seed for reproducibility (default: None)
    
    Returns:
        sampling_indicators: Binary array (1 if sampled) of shape (N,)
        inclusion_probs: Array of inclusion probabilities of shape (N,)
    
    Example:
        >>> xi, pi = classical_simple_random_sampling(N=1000, budget=0.1)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Equal inclusion probability for all units
    inclusion_probs = np.full(N, budget)
    
    # Independent Bernoulli sampling
    sampling_indicators = np.random.binomial(1, inclusion_probs)
    
    return sampling_indicators, inclusion_probs


def compute_sampling_probabilities(
    uncertainty: np.ndarray,
    budget: float,
    tau: float = 0.5
) -> np.ndarray:
    """
    Compute inclusion probabilities for active sampling.
    
    This is a utility function that computes the inclusion probabilities
    without performing the actual sampling.
    
    Args:
        uncertainty: Uncertainty estimates of shape (N,)
        budget: Sampling budget (target sampling rate)
        tau: Weight for uncertainty-based allocation (default: 0.5)
    
    Returns:
        inclusion_probs: Array of inclusion probabilities of shape (N,)
    
    Example:
        >>> pi = compute_sampling_probabilities(uncertainty, budget=0.1)
    """
    mean_uncertainty = uncertainty.mean()
    
    inclusion_probs = (
        tau * (uncertainty / mean_uncertainty) * budget +
        (1.0 - tau) * budget
    )
    
    # Ensure probabilities are in [0, 1]
    inclusion_probs = np.minimum(inclusion_probs, 1.0)
    
    return inclusion_probs
