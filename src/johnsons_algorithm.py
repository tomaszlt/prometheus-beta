import heapq
from typing import List, Dict, Union, Optional

def johnsons_algorithm(graph: Dict[int, Dict[int, int]]) -> Optional[List[List[Union[int, float]]]]:
    """
    Implement Johnson's algorithm to find the shortest paths between all pairs of vertices.
    
    Johnson's algorithm efficiently computes all-pairs shortest paths for sparse graphs.
    It uses Bellman-Ford to reweight edges and then Dijkstra's algorithm.
    
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
    
    # Get all vertices
    vertices = list(graph.keys())
    n = len(vertices)
    
    # Map vertices to 0-based indices if they're not already
    vertex_map = {v: i for i, v in enumerate(vertices)}
    reverse_map = {i: v for v, i in vertex_map.items()}
    
    # Add a dummy source vertex
    graph_with_dummy = graph.copy()
    dummy_vertex = max(vertices) + 1
    graph_with_dummy[dummy_vertex] = {v: 0 for v in vertices}
    
    # Step 1: Run Bellman-Ford from the dummy vertex to detect negative cycles
    # and compute vertex potentials
    def bellman_ford(graph, source):
        dist = {v: float('inf') for v in graph}
        dist[source] = 0
        
        # Relax edges |V| - 1 times
        for _ in range(len(graph) - 1):
            for u in graph:
                for v, weight in graph[u].items():
                    if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                        dist[v] = dist[u] + weight
        
        # Check for negative cycles
        for u in graph:
            for v, weight in graph[u].items():
                if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                    return None
        
        return dist
    
    # Compute potentials
    potentials = bellman_ford(graph_with_dummy, dummy_vertex)
    if potentials is None:
        return None  # Negative cycle detected
    
    # Remove the dummy vertex
    del graph_with_dummy[dummy_vertex]
    del potentials[dummy_vertex]
    
    # Step 2: Reweight edges
    reweighted_graph = {}
    for u in graph:
        reweighted_graph[u] = {}
        for v, weight in graph[u].items():
            reweighted_graph[u][v] = (
                weight + potentials[u] - potentials[v]
            )
    
    # Step 3: Run Dijkstra from each vertex
    def dijkstra(graph, source):
        dist = {v: float('inf') for v in graph}
        dist[source] = 0
        pq = [(0, source)]
        
        while pq:
            current_dist, u = heapq.heappop(pq)
            
            # If we've found a longer path, skip
            if current_dist > dist[u]:
                continue
            
            for v, weight in graph[u].items():
                distance = current_dist + weight
                
                if distance < dist[v]:
                    dist[v] = distance
                    heapq.heappush(pq, (distance, v))
        
        return dist
    
    # Compute shortest paths and restore original weights
    result = []
    for u in sorted(vertices):
        path_distances = dijkstra(reweighted_graph, u)
        
        # Restore original edge weights
        restored_distances = []
        for v in sorted(vertices):
            if path_distances[v] == float('inf'):
                restored_distances.append(float('inf'))
            else:
                restored_distances.append(
                    path_distances[v] - potentials[u] + potentials[v]
                )
        
        result.append(restored_distances)
    
    return result