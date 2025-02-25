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
    
    # Create a complete graph with all vertices
    complete_graph = {v: {} for v in vertices}
    for u in graph:
        for v, weight in graph[u].items():
            complete_graph[u][v] = weight
    
    # Add a dummy source vertex
    dummy_vertex = max(vertices) + 1
    complete_graph[dummy_vertex] = {v: 0 for v in vertices}
    
    # Step 1: Run Bellman-Ford from the dummy vertex to detect negative cycles
    def bellman_ford(graph, source):
        dist = {v: float('inf') for v in graph}
        dist[source] = 0
        
        # Relax edges |V| - 1 times
        for _ in range(len(graph) - 1):
            for u in graph:
                for v, weight in graph.get(u, {}).items():
                    if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                        dist[v] = dist[u] + weight
        
        # Check for negative cycles
        for u in graph:
            for v, weight in graph.get(u, {}).items():
                if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                    return None
        
        return dist
    
    # Compute potentials
    potentials = bellman_ford(complete_graph, dummy_vertex)
    if potentials is None:
        return None  # Negative cycle detected
    
    # Remove dummy vertex
    del complete_graph[dummy_vertex]
    del potentials[dummy_vertex]
    
    # Step 2: Reweight edges
    reweighted_graph = {}
    for u in complete_graph:
        reweighted_graph[u] = {}
        for v in complete_graph:
            # If no direct edge, skip
            if v not in complete_graph.get(u, {}):
                continue
            
            edge_weight = complete_graph[u][v]
            # Reweight the edge using vertex potentials
            reweighted_graph[u][v] = (
                edge_weight + potentials[u] - potentials[v]
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
            
            for v, weight in graph.get(u, {}).items():
                distance = current_dist + weight
                
                if distance < dist[v]:
                    dist[v] = distance
                    heapq.heappush(pq, (distance, v))
        
        return dist
    
    # Compute shortest paths and restore original weights
    result = []
    for u in vertices:
        path_distances = dijkstra(reweighted_graph, u)
        
        # Restore original edge weights
        restored_distances = []
        for v in vertices:
            # Default to infinity if no path exists
            if v not in path_distances or path_distances[v] == float('inf'):
                restored_distances.append(float('inf'))
            else:
                restored_distances.append(
                    path_distances[v] 
                    - potentials[u] 
                    + potentials[v]
                )
        
        result.append(restored_distances)
    
    return result