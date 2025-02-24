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
    
    # Transpose matrix_b for efficient column access
    transposed_b = {}
    for b_row, row_vector in matrix_b.items():
        for col, val in row_vector.items():
            if col not in transposed_b:
                transposed_b[col] = {}
            transposed_b[col][b_row] = val
    
    # Perform multiplication
    result = {}
    for a_row, a_row_vector in matrix_a.items():
        result_row = {}
        
        # Iterate through columns of transposed matrix_b
        for b_col, b_col_vector in transposed_b.items():
            # Compute dot product
            dot_product = sum(
                a_row_vector.get(k, 0) * b_col_vector.get(k, 0) 
                for k in set(a_row_vector) & set(b_col_vector)
            )
            
            # Only store non-zero values
            if dot_product != 0:
                result_row[b_col] = dot_product
        
        # Only store rows with non-zero values
        if result_row:
            result[a_row] = result_row
    
    return result