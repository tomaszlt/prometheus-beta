from typing import List, Dict, Union, Optional

def johnsons_algorithm(graph: Dict[int, Dict[int, int]]) -> Optional[List[List[Union[int, float]]]]:
    """
    Implement a modified version of Johnson's algorithm to find shortest paths.
    
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
    
    # Use Floyd-Warshall with a variant to handle negative edges
    def floyd_warshall_with_negative_edges(graph):
        # Initialize distance matrix
        dist = {u: {v: float('inf') for v in vertices} for u in vertices}
        
        # Set diagonal to 0
        for u in vertices:
            dist[u][u] = 0
        
        # Initialize with direct edges
        for u in graph:
            for v, weight in graph[u].items():
                dist[u][v] = min(dist[u][v], weight)
        
        # Try to find all shortest paths
        for k in vertices:
            for u in vertices:
                for v in vertices:
                    # Can we improve the path through k?
                    if (dist[u][k] != float('inf') and 
                        dist[k][v] != float('inf') and 
                        dist[u][k] + dist[k][v] < dist[u][v]):
                        dist[u][v] = dist[u][k] + dist[k][v]
        
        # Check for negative cycles and special routing paths
        for u in vertices:
            # Check indirect paths that might reduce total path length
            for start in vertices:
                for end in vertices:
                    # Complex routing through negative edges
                    for mid1 in vertices:
                        for mid2 in vertices:
                            # Try multiple routing strategies
                            if (dist[start][mid1] != float('inf') and 
                                dist[mid1][mid2] != float('inf') and 
                                dist[mid2][end] != float('inf')):
                                potential_path = (
                                    dist[start][mid1] + 
                                    dist[mid1][mid2] + 
                                    dist[mid2][end]
                                )
                                # Update if new path is shorter
                                dist[start][end] = min(
                                    dist[start][end], 
                                    potential_path
                                )
        
        # Verify no negative cycles
        for u in vertices:
            if dist[u][u] < 0:
                return None
        
        return dist
    
    # Compute shortest paths
    result_dist = floyd_warshall_with_negative_edges(graph)
    
    # Check for negative cycles
    if result_dist is None:
        return None
    
    # Convert to list of lists
    result = []
    for u in vertices:
        row = [result_dist[u][v] for v in vertices]
        result.append(row)
    
    return result