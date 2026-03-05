"""
Matrix Transpose

Problem: https://www.tensortonic.com/problems/matrix-transpose
"""

import numpy as np


def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    
    Mathematical definition: (A^T)[j, i] = A[i, j]
    An n×m matrix becomes an m×n matrix.
    
    Args:
        A: 2D NumPy array, shape (N, M) - input matrix
        
    Returns:
        New NumPy array of shape (M, N) - transposed matrix
    """
    A = np.asarray(A, dtype=float)
    
    N, M = A.shape
    
    # Create result array with swapped shape
    result = np.empty((M, N), dtype=float)
    
    # Manual indexing (requirement: no .T or np.transpose())
    for i in range(N):
        for j in range(M):
            result[j, i] = A[i, j]
    
    return result
