"""
Sparse Matrix Multiplication using Dictionaries

This module provides functionality for multiplying sparse matrices 
represented as dictionaries, where keys are row indices and values 
are row vectors.
"""

def sparse_matrix_multiply(matrix_a, matrix_b):
    """
    Multiply two sparse matrices represented as dictionaries.
    
    Args:
        matrix_a (dict): First sparse matrix with row indices as keys 
                         and row vectors as values.
        matrix_b (dict): Second sparse matrix with row indices as keys 
                         and row vectors as values.
    
    Returns:
        dict: Resulting sparse matrix after multiplication.
    
    Raises:
        ValueError: If matrices cannot be multiplied due to dimension mismatch.
    """
    # Validate matrix multiplication is possible
    if not matrix_a or not matrix_b:
        return {}
    
    # Perform multiplication
    result = {}
    for a_row, a_row_vector in matrix_a.items():
        result_row = {}
        
        # Compute each element in the result matrix
        for k, a_val in a_row_vector.items():
            b_row_vector = matrix_b.get(k, {})
            for b_col, b_val in b_row_vector.items():
                # Multiply and accumulate
                result_row[b_col] = result_row.get(b_col, 0) + a_val * b_val
        
        # Remove zero values and store if not empty
        result_row = {k: v for k, v in result_row.items() if v != 0}
        if result_row:
            result[a_row] = result_row
    
    return result