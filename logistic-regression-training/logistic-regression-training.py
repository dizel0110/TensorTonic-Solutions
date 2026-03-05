"""
Logistic Regression Training Loop

Problem: https://www.tensortonic.com/problems/logistic-regression-training-loop
"""

import numpy as np


def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))


def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    
    Model: p = σ(Xw + b)
    Loss: L = -1/N * Σ[y_i*log(p_i) + (1-y_i)*log(1-p_i)]
    
    Args:
        X: Input features, shape (N, D)
        y: Labels, shape (N,) with values 0 or 1
        lr: Learning rate (default 0.1)
        steps: Number of gradient descent steps (default 1000)
        
    Returns:
        Tuple (w, b) where:
            w: Weights, shape (D,)
            b: Bias, scalar float
    """
    # Convert inputs to numpy arrays
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    
    N, D = X.shape
    
    # Initialize weights and bias to zeros
    w = np.zeros(D)
    b = 0.0
    
    for _ in range(steps):
        # Forward pass: compute predictions
        z = X @ w + b  # Linear combination, shape (N,)
        p = _sigmoid(z)  # Predictions (probabilities), shape (N,)
        
        # Compute gradients
        error = p - y  # shape (N,)
        dw = (X.T @ error) / N  # Gradient for weights, shape (D,)
        db = np.mean(error)  # Gradient for bias, scalar
        
        # Update parameters via gradient descent
        w = w - lr * dw
        b = b - lr * db
    
    return (w, b)
