"""
Create Voronoi diagram from Point geometries in a GeoDataFrame.

Usage:
    from voronoi_from_points import create_voronoi
    voronoi_gdf = create_voronoi(df_boul, clip_to=None)
    voronoi_gdf.plot()
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt


def create_voronoi(gdf_points, clip_to=None):
    """
    Create Voronoi diagram from Point geometries.
    
    Args:
        gdf_points: GeoDataFrame with Point geometries
        clip_to: Optional GeoDataFrame/Polygon to clip Voronoi regions to.
                 If None, uses bounding box of points
    
    Returns:
        GeoDataFrame with Voronoi polygons
    """
    # Extract coordinates from Point geometries
    points = np.array([[geom.x, geom.y] for geom in gdf_points.geometry])
    
    # Create Voronoi diagram
    vor = Voronoi(points)
    
    # Create bounding box for clipping infinite regions
    if clip_to is None:
        # Use bounds of the points
        minx, miny = points.min(axis=0)
        maxx, maxy = points.max(axis=0)
        # Add some padding
        padding = max((maxx - minx), (maxy - miny)) * 0.1
        bbox = Polygon([
            [minx - padding, miny - padding],
            [maxx + padding, miny - padding],
            [maxx + padding, maxy + padding],
            [minx - padding, maxy + padding]
        ])
    else:
        # Use provided clipping geometry
        if isinstance(clip_to, gpd.GeoDataFrame):
            bbox = clip_to.unary_union if len(clip_to) > 1 else clip_to.geometry.iloc[0]
        else:
            bbox = clip_to
    
    # Convert Voronoi regions to polygons
    voronoi_polygons = []
    
    for idx, point_idx in enumerate(vor.point_region):
        region = vor.regions[point_idx]
        
        # Skip empty regions or regions with -1 (infinite)
        if -1 in region or len(region) == 0:
            # Handle infinite regions by clipping
            vertices = []
            for vertex_idx in region:
                if vertex_idx == -1:
                    continue
                vertices.append(vor.vertices[vertex_idx])
            
            if len(vertices) >= 3:
                # Create polygon and clip to bounding box
                poly = Polygon(vertices)
                if poly.is_valid:
                    clipped = poly.intersection(bbox)
                    if not clipped.is_empty:
                        voronoi_polygons.append(clipped)
                else:
                    voronoi_polygons.append(None)
            else:
                voronoi_polygons.append(None)
        else:
            # Finite region
            vertices = [vor.vertices[i] for i in region]
            if len(vertices) >= 3:
                poly = Polygon(vertices)
                # Clip to bounding box
                clipped = poly.intersection(bbox)
                if not clipped.is_empty and clipped.is_valid:
                    voronoi_polygons.append(clipped)
                else:
                    voronoi_polygons.append(None)
            else:
                voronoi_polygons.append(None)
    
    # Create GeoDataFrame
    voronoi_gdf = gpd.GeoDataFrame(
        geometry=[poly for poly in voronoi_polygons if poly is not None],
        crs=gdf_points.crs
    )
    
    # Copy attributes from original points if they exist
    if len(voronoi_gdf) == len(gdf_points):
        for col in gdf_points.columns:
            if col != 'geometry':
                voronoi_gdf[col] = gdf_points[col].values
    
    return voronoi_gdf


def plot_voronoi(gdf_points, voronoi_gdf, clip_to=None, ax=None, **plot_kwargs):
    """
    Plot Voronoi diagram with original points.
    
    Args:
        gdf_points: GeoDataFrame with Point geometries
        voronoi_gdf: GeoDataFrame with Voronoi polygons
        clip_to: Optional geometry to clip plot to
        ax: Matplotlib axis (creates new one if None)
        **plot_kwargs: Additional arguments for plotting
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot Voronoi regions
    voronoi_gdf.plot(ax=ax, edgecolor='black', facecolor='lightblue', 
                     alpha=0.5, linewidth=1, **plot_kwargs)
    
    # Plot original points
    gdf_points.plot(ax=ax, color='red', markersize=50, zorder=5)
    
    # Clip plot if requested
    if clip_to is not None:
        if isinstance(clip_to, gpd.GeoDataFrame):
            clip_to.plot(ax=ax, facecolor='none', edgecolor='gray', 
                        linewidth=2, linestyle='--')
        ax.set_xlim(*clip_to.bounds[[0, 2]])
        ax.set_ylim(*clip_to.bounds[[1, 3]])
    
    ax.set_aspect('equal')
    ax.set_title('Voronoi Diagram')
    plt.tight_layout()
    
    return ax


# Example usage:
if __name__ == "__main__":
    # Example with sample points
    # Replace df_boul with your actual GeoDataFrame
    # df_boul = gpd.read_file('your_points.shp')
    
    # Create Voronoi diagram
    # voronoi_gdf = create_voronoi(df_boul)
    
    # Plot it
    # ax = plot_voronoi(df_boul, voronoi_gdf)
    # plt.show()
    
    # Or clip to a specific region
    # boundary = gpd.read_file('boundary.shp')
    # voronoi_clipped = create_voronoi(df_boul, clip_to=boundary)
    # ax = plot_voronoi(df_boul, voronoi_clipped, clip_to=boundary)
    # plt.show()
    
    pass

