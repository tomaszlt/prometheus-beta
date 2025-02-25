import pytest
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from johnsons_algorithm import johnsons_algorithm

def test_simple_graph():
    """Test a simple graph with positive weights"""
    graph = {
        0: {1: 5, 2: 2},
        1: {2: 1},
        2: {1: -3}
    }
    
    result = johnsons_algorithm(graph)
    assert result is not None
    
    # Expected shortest paths for this specific graph
    assert len(result) == 3
    assert result[0][0] == 0  # Distance to self is 0
    assert result[0][1] == 4  # Shortest path from 0 to 1 
    assert result[0][2] == 2  # Shortest path from 0 to 2

def test_disconnected_graph():
    """Test a graph with disconnected vertices"""
    graph = {
        0: {1: 5},
        2: {3: 3}
    }
    
    result = johnsons_algorithm(graph)
    assert result is not None
    
    # Check infinity for unreachable paths
    assert result[0][2] == float('inf')
    assert result[2][0] == float('inf')

def test_negative_edge_weights():
    """Test graph with some negative edge weights"""
    graph = {
        0: {1: 6, 2: 3},
        1: {2: -2},
        2: {1: 7}
    }
    
    result = johnsons_algorithm(graph)
    assert result is not None
    
    # Verify some known shortest paths
    assert result[0][1] == 3  # 0 -> 1 via 2
    assert result[0][2] == 3  # 0 -> 2 direct

def test_negative_cycle():
    """Test graph with a negative cycle"""
    graph = {
        0: {1: 1},
        1: {2: -3},
        2: {0: -2}
    }
    
    result = johnsons_algorithm(graph)
    assert result is None  # Should detect and return None

def test_empty_graph():
    """Test handling of empty graph"""
    with pytest.raises(ValueError):
        johnsons_algorithm({})

def test_single_vertex_graph():
    """Test graph with a single vertex"""
    graph = {0: {}}
    
    result = johnsons_algorithm(graph)
    assert result is not None
    assert len(result) == 1
    assert result[0][0] == 0  # Distance to self is 0

def test_complete_graph():
    """Test a fully connected graph"""
    graph = {
        0: {1: 1, 2: 4, 3: 3},
        1: {0: 1, 2: 2, 3: 5},
        2: {0: 4, 1: 2, 3: 1},
        3: {0: 3, 1: 5, 2: 1}
    }
    
    result = johnsons_algorithm(graph)
    assert result is not None
    
    # Some specific path length checks
    assert result[0][1] == 1
    assert result[1][2] == 2
    assert result[2][3] == 1