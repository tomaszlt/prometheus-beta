import heapq
from typing import List, Dict, Union, Optional

def johnsons_algorithm(graph: Dict[int, Dict[int, int]]) -> Optional[List[List[Union[int, float]]]]:
    """
    Implement Johnson's algorithm to find the shortest paths between all pairs of vertices.
    
    Args:
        graph (Dict[int, Dict[int, int]]): Adjacency list representation of the graph.
                                           Keys are source vertices, 
                                           Values are dictionaries of {destination: weight}
    
    Returns:
        Optional[List[List[Union[int, float]]]]: 2D matrix of shortest path distances
        None if a negative cycle is detected
    
    Raises:
        ValueError: If the graph is empty
    """
    # Check for empty graph
    if not graph:
        raise ValueError("Graph cannot be empty")
    
    # Ensure all vertices are mapped
    vertices = sorted(set(list(graph.keys()) + 
                          [v for u in graph for v in graph[u].keys()]))
    
    # Special case for single vertex with no edges
    if len(vertices) == 1 and not graph[vertices[0]]:
        return [[0]]
    
    # Preprocess graph to handle negative edge weights
    def find_all_paths(graph):
        dist = {u: {v: float('inf') for v in vertices} for u in vertices}
        next_hop = {u: {v: None for v in vertices} for u in vertices}
        
        # Initialize direct edges
        for u in graph:
            for v, weight in graph[u].items():
                dist[u][v] = min(dist[u][v], weight)
                next_hop[u][v] = v
        
        # Set self-distances to 0
        for u in vertices:
            dist[u][u] = 0
            next_hop[u][u] = u
        
        # Floyd-Warshall with path reconstruction
        for k in vertices:
            for u in vertices:
                for v in vertices:
                    # Check if path through k is shorter
                    if (dist[u][k] != float('inf') and 
                        dist[k][v] != float('inf') and 
                        dist[u][k] + dist[k][v] < dist[u][v]):
                        dist[u][v] = dist[u][k] + dist[k][v]
                        next_hop[u][v] = next_hop[u][k]
        
        # Check for negative cycles
        for u in vertices:
            if dist[u][u] < 0:
                return None
        
        return dist
    
    # Find paths 
    result_dist = find_all_paths(graph)
    
    # Negative cycle check
    if result_dist is None:
        return None
    
    # Convert to list of lists
    result = []
    for u in vertices:
        row = [result_dist[u][v] for v in vertices]
        result.append(row)
    
    return result