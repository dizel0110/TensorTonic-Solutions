"""
Positional Encoding (sin/cos)

Problem: https://www.tensortonic.com/problems/positional-encoding
"""

import numpy as np


def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    
    Mathematical definition (Attention Is All You Need):
    PE(pos, 2i) = sin(pos / base^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / base^(2i/d_model))
    
    Even-indexed columns use sine, odd-indexed columns use cosine.
    Frequency decreases with dimension index.
    
    Args:
        seq_len: Length of sequence (T)
        d_model: Model dimension (d)
        base: Base for frequency calculation (default 10000.0)
        
    Returns:
        np.ndarray of shape (seq_len, d_model), dtype=float
    """
    # Create position vector: shape (seq_len, 1)
    positions = np.arange(seq_len, dtype=float).reshape(-1, 1)
    
    # Create frequency vector for i values: shape (ceil(d_model/2),)
    # i represents the column pair index (sin + cos pair)
    num_pairs = (d_model + 1) // 2  # ceil(d_model / 2)
    i = np.arange(num_pairs, dtype=float)
    
    # Compute divisor: base^(2i/d_model)
    # This creates decreasing frequencies for higher dimensions
    divisors = base ** (2 * i / d_model)
    
    # Compute angles: pos / divisor
    # Broadcasting: (seq_len, 1) / (num_pairs,) -> (seq_len, num_pairs)
    angles = positions / divisors
    
    # Compute sin and cos
    sin_vals = np.sin(angles)  # (seq_len, num_pairs)
    cos_vals = np.cos(angles)  # (seq_len, num_pairs)
    
    # Stack sin and cos alternately
    # Result shape: (seq_len, 2 * num_pairs)
    pe = np.empty((seq_len, 2 * num_pairs), dtype=float)
    pe[:, 0::2] = sin_vals  # Even columns: sin
    pe[:, 1::2] = cos_vals  # Odd columns: cos
    
    # If d_model is odd, we have one extra column (2*num_pairs = d_model + 1)
    # Trim to exact d_model (last column is sin, as required)
    if pe.shape[1] > d_model:
        pe = pe[:, :d_model]
    
    return pe
