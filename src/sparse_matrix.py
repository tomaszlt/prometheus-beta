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
        for b_col in set().union(*(vector.keys() for vector in matrix_b.values())):
            # Compute dot product for this column
            dot_product = sum(
                a_row_vector.get(k, 0) * matrix_b.get(k, {}).get(b_col, 0)
                for k in a_row_vector.keys()
            )
            
            # Only store non-zero values
            if dot_product != 0:
                result_row[b_col] = dot_product
        
        # Only store rows with non-zero values
        if result_row:
            result[a_row] = result_row
    
    return result