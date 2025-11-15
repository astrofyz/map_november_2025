"""
Find shortest path between points using NetworkX and extract only the segment.

Requirements:
    pip install networkx shapely geopandas
"""

import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, LineString
import numpy as np


def lines_to_graph(lines_gdf):
    """
    Convert LineString GeoDataFrame to NetworkX graph.
    
    Args:
        lines_gdf: GeoDataFrame with LineString geometries
    
    Returns:
        NetworkX graph and mapping of edges to geometries
    """
    G = nx.Graph()
    edge_geometries = {}
    
    for idx, row in lines_gdf.iterrows():
        line = row.geometry
        coords = list(line.coords)
        
        # Add nodes and edges
        for i in range(len(coords) - 1):
            node1 = coords[i]
            node2 = coords[i + 1]
            
            # Calculate edge weight (distance)
            edge_length = Point(node1).distance(Point(node2))
            
            # Add edge with weight
            G.add_edge(node1, node2, weight=edge_length, length=edge_length)
            
            # Store geometry for this segment
            edge_geometries[(node1, node2)] = LineString([node1, node2])
            edge_geometries[(node2, node1)] = LineString([node2, node1])  # Bidirectional
    
    return G, edge_geometries


def find_closest_node(graph, point, tolerance=0.001):
    """
    Find the closest node in the graph to a given point.
    
    Args:
        graph: NetworkX graph
        point: Point geometry or (x, y) tuple
        tolerance: Maximum distance to consider
    
    Returns:
        Closest node coordinates
    """
    if isinstance(point, Point):
        target = (point.x, point.y)
    else:
        target = point
    
    # Get all nodes
    nodes = list(graph.nodes())
    
    # Find closest node
    min_dist = float('inf')
    closest_node = None
    
    for node in nodes:
        dist = np.sqrt((node[0] - target[0])**2 + (node[1] - target[1])**2)
        if dist < min_dist:
            min_dist = dist
            closest_node = node
    
    if min_dist > tolerance:
        # If no close node, add the point to the graph temporarily
        closest_node = target
    
    return closest_node


def find_shortest_path_between_points(lines_gdf, start_point, end_point):
    """
    Find shortest path between two points using the line network.
    
    Args:
        lines_gdf: GeoDataFrame with LineString geometries
        start_point: Start Point geometry or (x, y)
        end_point: End Point geometry or (x, y)
    
    Returns:
        LineString geometry of the shortest path segment
    """
    # Convert lines to graph
    G, edge_geometries = lines_to_graph(lines_gdf)
    
    # Find closest nodes
    start_node = find_closest_node(G, start_point)
    end_node = find_closest_node(G, end_point)
    
    # If nodes don't exist, add them and connect to nearest edges
    if start_node not in G:
        # Find closest edge and snap to it
        start_node = find_closest_node(G, start_point)
        if start_node not in G:
            # Add node and connect to nearest nodes
            nearest = min(G.nodes(), key=lambda n: Point(start_point).distance(Point(n)))
            G.add_node(start_node)
            dist = Point(start_point).distance(Point(nearest))
            G.add_edge(start_node, nearest, weight=dist, length=dist)
    
    if end_node not in G:
        nearest = min(G.nodes(), key=lambda n: Point(end_point).distance(Point(n)))
        G.add_node(end_node)
        dist = Point(end_point).distance(Point(nearest))
        G.add_edge(end_node, nearest, weight=dist, length=dist)
    
    # Find shortest path
    try:
        path_nodes = nx.shortest_path(G, start_node, end_node, weight='weight')
    except nx.NetworkXNoPath:
        print(f"No path found between points")
        return None
    
    # Reconstruct LineString from path
    path_coords = list(path_nodes)
    path_line = LineString(path_coords)
    
    return path_line


def extract_segment_between_points(line, start_point, end_point, tolerance=0.001):
    """
    Extract only the segment of a line between two points.
    
    Args:
        line: LineString geometry
        start_point: Start Point
        end_point: End Point
        tolerance: Distance tolerance for finding points on line
    
    Returns:
        LineString segment between points
    """
    from shapely.ops import substring
    
    # Project points onto line
    start_proj = line.interpolate(line.project(start_point))
    end_proj = line.interpolate(line.project(end_point))
    
    # Get distances along line
    start_dist = line.project(start_point)
    end_dist = line.project(end_point)
    
    # Ensure start < end
    if start_dist > end_dist:
        start_dist, end_dist = end_dist, start_dist
    
    # Extract segment
    segment = substring(line, start_dist, end_dist)
    
    return segment


def find_best_route_and_segment(lines_gdf, start_point, end_point):
    """
    Find the shortest route and extract only the segment between points.
    
    Args:
        lines_gdf: GeoDataFrame with LineString geometries
        start_point: Start Point geometry
        end_point: End Point geometry
    
    Returns:
        LineString of the shortest route segment between points
    """
    # Method 1: Use NetworkX to find shortest path
    shortest_path = find_shortest_path_between_points(lines_gdf, start_point, end_point)
    
    if shortest_path is not None:
        return shortest_path
    
    # Method 2: Fallback - find lines that contain both points, then extract segment
    lines_with_points = gpd.sjoin(
        lines_gdf,
        gpd.GeoDataFrame(geometry=[start_point, end_point]),
        how='inner',
        predicate='intersects'
    )
    
    if len(lines_with_points) == 0:
        return None
    
    # Find shortest line that contains both points
    best_line = None
    min_length = float('inf')
    
    for idx, row in lines_with_points.iterrows():
        line = row.geometry
        # Check if line contains both points
        if (line.distance(start_point) < 0.001 and 
            line.distance(end_point) < 0.001):
            # Extract segment
            segment = extract_segment_between_points(line, start_point, end_point)
            if segment.length < min_length:
                min_length = segment.length
                best_line = segment
    
    return best_line


# Example usage:
if __name__ == "__main__":
    # Example:
    # lines_gdf = gpd.read_file('lines.shp')
    # start_point = Point(2.3, 48.8)
    # end_point = Point(2.4, 48.9)
    # 
    # shortest_segment = find_best_route_and_segment(lines_gdf, start_point, end_point)
    # 
    # # Plot
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(12, 8))
    # lines_gdf.plot(ax=ax, color='lightgray', alpha=0.3)
    # gpd.GeoDataFrame(geometry=[shortest_segment]).plot(ax=ax, color='red', linewidth=2)
    # gpd.GeoDataFrame(geometry=[start_point, end_point]).plot(ax=ax, color='blue', markersize=100)
    # plt.show()
    
    pass

