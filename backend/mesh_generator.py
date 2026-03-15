import os
import argparse
import numpy as np
import trimesh

def point_cloud_to_mesh(glb_path: str, output_path: str, depth=8):
    """
    Reads a .glb point cloud, estimates normals, and generates a solid mesh
    using Open3D's Poisson Surface Reconstruction.
    """
    print(f"Loading Point Cloud from {glb_path}...")
    
    # 1. Load the raw points using trimesh (since the GLB is just a trimesh PointCloud export)
    scene = trimesh.load(glb_path, force='scene')
    
    # Extract vertices and colors from the nested scene graph
    points = []
    colors = []
    for name, geom in scene.geometry.items():
        if hasattr(geom, 'vertices') and hasattr(geom, 'visual') and hasattr(geom.visual, 'vertex_colors') and geom.visual.vertex_colors is not None:
            color_array = np.array(geom.visual.vertex_colors)
            if len(color_array) == len(geom.vertices):
                points.append(np.array(geom.vertices))
                # trimesh colors are usually RGBA 0-255
                colors.append(color_array[:, :3] / 255.0)
            
    if not points:
        raise ValueError("No points found in the GLB file!")
        
    pts = np.vstack(points)
    cols = np.vstack(colors) if colors else np.zeros_like(pts)
    
    print(f"Extracted {len(pts)} points. Initializing Open3D processing...")
    
    # We import open3d here inside the function to ensure it's gracefully handled if not installed
    import open3d as o3d
    
    # 2. Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    
    # 2.5 Remove extreme noise/outliers that break the Poisson algorithm
    print("Removing statistical outliers...")
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.5)
    
    # 3. Estimate Normals
    print("Estimating surface normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)
    
    # 4. Ball Pivoting Algorithm (BPA) Reconstruction
    print("Running Ball Pivoting Algorithm (BPA) Surface Reconstruction...")
    # Calculate average distance between points to determine the ball radii
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    print(f"Average point distance: {avg_dist:.4f}")
    
    # Use multiples of the average distance to fill holes of various sizes
    radii = [avg_dist * 1.5, avg_dist * 3.0, avg_dist * 6.0]
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    
    # 4.5 Transfer colors from the original point cloud to the new mesh vertices!
    print("Transferring colors from point cloud to mesh (Vectorized via SciPy)...")
    from scipy.spatial import cKDTree
    
    mesh_vertices = np.asarray(mesh.vertices)
    pcd_colors = np.asarray(pcd.colors)
    pcd_vertices = np.asarray(pcd.points)
    
    # Build KDTree on the original point cloud geometry
    kdtree = cKDTree(pcd_vertices)
    
    # Query the closest original point for EVERY new mesh vertex simultaneously
    distances, indices = kdtree.query(mesh_vertices, k=1, workers=-1)
    
    # Map the colors and apply them
    mesh_colors = pcd_colors[indices]
    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)

    # 5. Smooth the mesh to make it look nicer
    print("Smoothing mesh...")
    mesh = mesh.filter_smooth_simple(number_of_iterations=1)
    mesh.compute_vertex_normals()

    # 7. Export the final mesh
    print(f"Exporting solid Mesh with colors to {output_path}...")
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    vertex_colors = np.asarray(mesh.vertex_colors)
    
    if vertex_colors.shape[0] > 0:
        vertex_colors_255 = np.clip(vertex_colors * 255, 0, 255).astype(np.uint8)
        vertex_colors_rgba = np.hstack([vertex_colors_255, np.full((vertex_colors.shape[0], 1), 255, dtype=np.uint8)])
    else:
        vertex_colors_rgba = None
        
    out_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors_rgba)
    out_mesh.export(output_path)
    print("Mesh generation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Point Cloud to Mesh")
    parser.add_argument("--input", type=str, required=True, help="Input .glb Point Cloud")
    parser.add_argument("--output", type=str, required=True, help="Output .glb Mesh")
    parser.add_argument("--depth", type=int, default=8, help="Poisson tree depth (6 to 10)")
    args = parser.parse_args()
    
    point_cloud_to_mesh(args.input, args.output, args.depth)
