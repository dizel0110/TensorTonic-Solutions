"""
Pad Sequences

Problem: https://www.tensortonic.com/problems/pad-sequences
"""

import numpy as np


def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Pad sequences to equal length.
    
    Args:
        seqs: List of sequences (lists of ints)
        pad_value: Value for padding (default 0)
        max_len: Maximum length (default: max length in seqs)
        
    Returns:
        np.ndarray of shape (N, L) where:
            N = len(seqs)
            L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Handle empty input
    if len(seqs) == 0:
        return np.array([], dtype=int).reshape(0, 0)
    
    # Compute max_len if not provided
    if max_len is None:
        max_len = max(len(seq) for seq in seqs) if seqs else 0
    
    # Handle case where max_len is 0
    if max_len == 0:
        return np.array([], dtype=int).reshape(len(seqs), 0)
    
    N = len(seqs)
    
    # Initialize result array with pad_value
    result = np.full((N, max_len), pad_value, dtype=int)
    
    # Copy each sequence (with truncation if needed)
    for i, seq in enumerate(seqs):
        seq_len = min(len(seq), max_len)
        if seq_len > 0:
            result[i, :seq_len] = seq[:seq_len]
    
    return result
