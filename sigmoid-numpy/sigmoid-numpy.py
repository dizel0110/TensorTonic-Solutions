"""
Sigmoid function implementation using NumPy

Problem: https://www.tensortonic.com/problems/sigmoid-numpy
"""

import numpy as np


def sigmoid(x) -> np.ndarray:
    """
    Compute the sigmoid function for input.
    
    Sigmoid formula: σ(x) = 1 / (1 + e^(-x))
    
    Args:
        x: Scalar, list, or NumPy array
        
    Returns:
        NumPy array of floats with sigmoid values
    """
    x = np.asarray(x, dtype=float)
    return np.asarray(1 / (1 + np.exp(-x)))
