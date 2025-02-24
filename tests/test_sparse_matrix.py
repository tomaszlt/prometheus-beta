"""
Tests for sparse matrix multiplication module.
"""

import pytest
from src.sparse_matrix import sparse_matrix_multiply

def test_basic_multiplication():
    """Test basic sparse matrix multiplication."""
    # Matrix A: 2x3 sparse matrix
    # [1 0 2]
    # [0 3 0]
    matrix_a = {
        0: {0: 1, 2: 2},
        1: {1: 3}
    }
    
    # Matrix B: 3x2 sparse matrix
    # [1 0]
    # [2 3]
    # [0 4]
    matrix_b = {
        0: {0: 1},
        1: {1: 3},
        2: {1: 4}
    }
    
    # Expected result: 2x2 sparse matrix
    # [1*1 + 2*2  1*0 + 2*3]
    # [3*2       3*3]
    expected = {
        0: {0: 5, 1: 6},
        1: {0: 6, 1: 9}
    }
    
    assert sparse_matrix_multiply(matrix_a, matrix_b) == expected

def test_empty_matrices():
    """Test multiplication with empty matrices."""
    assert sparse_matrix_multiply({}, {}) == {}
    assert sparse_matrix_multiply({0: {}}, {0: {}}) == {}

def test_zero_matrices():
    """Test multiplication with zero matrices."""
    matrix_a = {0: {0: 0}, 1: {1: 0}}
    matrix_b = {0: {0: 0}, 1: {1: 0}}
    
    assert sparse_matrix_multiply(matrix_a, matrix_b) == {}

def test_sparse_identity_matrix():
    """Test multiplication with sparse identity matrix."""
    matrix = {
        0: {0: 5, 2: 7},
        1: {1: 3},
        2: {0: 1, 2: 4}
    }
    
    identity = {
        0: {0: 1},
        1: {1: 1},
        2: {2: 1}
    }
    
    assert sparse_matrix_multiply(matrix, identity) == matrix
    assert sparse_matrix_multiply(identity, matrix) == matrix

def test_complex_sparse_multiplication():
    """Test multiplication with more complex sparse matrices."""
    matrix_a = {
        0: {1: 2, 3: 5},
        2: {0: 1, 2: 3}
    }
    
    matrix_b = {
        1: {0: 4, 2: 6},
        3: {1: 7, 3: 8}
    }
    
    expected = {
        0: {0: 8, 2: 12},
        2: {1: 21, 3: 24}
    }
    
    assert sparse_matrix_multiply(matrix_a, matrix_b) == expected

def test_large_sparse_matrices():
    """Test multiplication with large sparse matrices."""
    # Create large sparse matrices
    matrix_a = {
        i: {j: i*j for j in range(0, 1000, 10)} 
        for i in range(0, 100, 20)
    }
    
    matrix_b = {
        j: {i: j*i for i in range(0, 100, 20)} 
        for j in range(0, 1000, 10)
    }
    
    result = sparse_matrix_multiply(matrix_a, matrix_b)
    
    # Verify result is a sparse matrix with expected structure
    assert isinstance(result, dict)
    for row in result:
        assert isinstance(result[row], dict)
        for col in result[row]:
            assert result[row][col] != 0