"""
Gradient Descent for 1D Quadratic

Problem: https://www.tensortonic.com/problems/gradient-descent-quadratic
"""

import numpy as np


def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Minimize f(x) = ax² + bx + c using gradient descent.
    
    Gradient: f'(x) = 2ax + b
    
    Update rule: x = x - lr * f'(x)
               = x - lr * (2ax + b)
    
    Args:
        a: Coefficient of x² (must be > 0)
        b: Coefficient of x
        c: Constant term
        x0: Initial x value
        lr: Learning rate (must be > 0)
        steps: Number of iterations (must be >= 1)
        
    Returns:
        Final x value as Python float
    """
    x = float(x0)
    
    for _ in range(steps):
        # Compute gradient: f'(x) = 2ax + b
        gradient = 2 * a * x + b
        
        # Update x: x = x - lr * gradient
        x = x - lr * gradient
    
    return float(x)
