"""Point-cloud -> mesh, with four algorithm choices.

Four algorithms are exposed via ``--algo``:

    poisson  (default) -- Open3D's Poisson Surface Reconstruction.
        Smooth, watertight, but inherently does some smoothing as part of
        the implicit-field iso-extraction. Defaults here are tuned for
        fidelity (rigid fit) over smoothness, but Poisson can never produce
        a mesh whose vertices exactly equal the input points.

    bpa      -- Ball Pivoting Algorithm. Maximally rigid: mesh vertices
        ARE the input points (no movement, no smoothing whatsoever).
        Triangles are drawn where a virtual ball of a given radius can
        rest on three nearby points. Produces holes wherever the cloud
        is too sparse for the ball to bridge. No density culling needed.

    alpha    -- Alpha Shape. A middle ground between Poisson and BPA.
        Produces a tighter, more detailed surface than BPA (fewer holes)
        without the global smoothing/watertightness of Poisson. Good for
        dense clouds where you want to preserve sharp edges and avoid
        the "balloon" effect.

    planes   -- RANSAC plane detection + convex hull triangulation.
        Detects dominant planes (walls, floor, ceiling) and meshes each
        as a flat polygon. Remaining non-plane points are meshed with BPA.
        Best for indoor / room scenes with clear planar structure.

If Poisson is "still too smooth" or producing "weird bumps", try alpha, bpa,
or planes.

Tuning Poisson:

    Too smooth still         -> --depth 11 --point_weight 20 --density_quantile 0.5
    Swiss cheese (too sparse) -> --density_quantile 0.15 --smooth_iters 1

Tuning BPA:

    Too many holes           -> --bpa_radius_mult 3.0 (or 4.0)
    Bridging across gaps     -> --bpa_radius_mult 1.5

Tuning Alpha:

    Too fragmented (holes)   -> --alpha_mult 3.0 (or 4.0)
    Too blobby (fills gaps)  -> --alpha_mult 1.0

Tuning Planes:

    Missing small planes     -> --plane_min_plane_ratio 0.02
    Too many tiny planes     -> --plane_min_plane_ratio 0.10
    Gaps in walls            -> --plane_distance_threshold 0.03 (more lenient)
"""

import argparse
import logging

import numpy as np
import trimesh

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_pointcloud(glb_path: str):
    """Read a colored ``.glb`` point cloud and return (points, colors) numpy arrays."""
    logger.info(f"Loading point cloud from {glb_path}...")
    scene = trimesh.load(glb_path, force="scene")

    points: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    for _name, geom in scene.geometry.items():
        if (
            hasattr(geom, "vertices")
            and hasattr(geom, "visual")
            and hasattr(geom.visual, "vertex_colors")
            and geom.visual.vertex_colors is not None
        ):
            color_array = np.array(geom.visual.vertex_colors)
            if len(color_array) == len(geom.vertices):
                points.append(np.array(geom.vertices))
                colors.append(color_array[:, :3] / 255.0)

    if not points:
        raise ValueError("No points found in the GLB file!")

    pts = np.vstack(points)
    cols = np.vstack(colors) if colors else np.zeros_like(pts)
    logger.info(f"Extracted {len(pts):,} points.")
    return pts, cols


def _build_pcd(pts, cols, outlier_nb, outlier_std, normal_knn, orient_normals_camera):
    """Build an Open3D PointCloud with outlier removal + oriented normals."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)

    logger.info(
        f"Removing statistical outliers (nb={outlier_nb}, std={outlier_std})..."
    )
    pcd, _ind = pcd.remove_statistical_outlier(
        nb_neighbors=outlier_nb, std_ratio=outlier_std
    )
    logger.info(f"  {len(pcd.points):,} points remain after outlier removal.")

    logger.info(f"Estimating surface normals (KNN={normal_knn})...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=normal_knn))

    if orient_normals_camera is not None:
        logger.info(
            f"Orienting normals toward camera at {orient_normals_camera}..."
        )
        pcd.orient_normals_towards_camera_location(
            camera_location=np.asarray(orient_normals_camera, dtype=np.float64)
        )
    else:
        logger.info("Orienting normals via consistent tangent plane (k=30)...")
        pcd.orient_normals_consistent_tangent_plane(k=30)

    return pcd


def _run_poisson(
    pcd, depth, scale, linear_fit, point_weight, density_quantile
):
    """Run Poisson, then density-cull. Returns the Open3D TriangleMesh."""
    import open3d as o3d

    logger.info(
        f"Running Poisson Surface Reconstruction "
        f"(depth={depth}, scale={scale}, linear_fit={linear_fit}, "
        f"point_weight={point_weight})..."
    )
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=depth,
            width=0,
            scale=scale,
            linear_fit=linear_fit,
            point_weight=point_weight,
        )
    except TypeError:
        logger.warning(
            "This Open3D version does not accept point_weight=; "
            "falling back to default (=4). Mesh will be slightly less rigid."
        )
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=0, scale=scale, linear_fit=linear_fit
        )

    densities = np.asarray(densities)
    if density_quantile > 0:
        thr = float(np.quantile(densities, density_quantile))
        low_density = densities < thr
        n_drop = int(low_density.sum())
        logger.info(
            f"Density culling: dropping {n_drop:,}/{len(densities):,} vertices "
            f"below quantile {density_quantile:.2f} (= density {thr:.4g})."
        )
        mesh.remove_vertices_by_mask(low_density)

    return mesh


def _run_bpa(pcd, bpa_radius_mult, bpa_radii_levels):
    """Run Ball Pivoting Algorithm. Returns the Open3D TriangleMesh.

    The radii are chosen as geometric multiples of the average nearest-neighbor
    distance of the (cleaned) point cloud, which makes the algorithm
    scale-invariant. Multiple radii lets the ball "fall through" small gaps
    at one scale and "rest" on wider surface chunks at another -- BPA's
    standard recipe for handling non-uniform sampling density.
    """
    import open3d as o3d

    logger.info("Computing average nearest-neighbor distance for BPA radii...")
    nn_dists = np.asarray(pcd.compute_nearest_neighbor_distance())
    avg_d = float(np.mean(nn_dists))
    logger.info(f"  avg_nn_dist = {avg_d:.6g}")

    # radii: avg_d * mult, * 2*mult, * 4*mult, ...  (geometric progression)
    radii = [avg_d * bpa_radius_mult * (2 ** i) for i in range(bpa_radii_levels)]
    logger.info(f"Running BPA with radii = {radii}")

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    logger.info(
        f"BPA produced {len(mesh.vertices):,} vertices and "
        f"{len(mesh.triangles):,} triangles."
    )
    return mesh


def _run_alpha(pcd, alpha_mult):
    """Run Alpha Shape. Returns the Open3D TriangleMesh.

    Alpha shapes are a generalization of the convex hull. Small alpha = tight
    fit (many holes), large alpha = fills gaps (may bridge discontinuities).
    We auto-estimate alpha from the point cloud's nearest-neighbor distance.
    """
    import open3d as o3d

    logger.info("Computing average nearest-neighbor distance for alpha...")
    nn_dists = np.asarray(pcd.compute_nearest_neighbor_distance())
    avg_d = float(np.mean(nn_dists))
    alpha = avg_d * alpha_mult
    logger.info(f"  avg_nn_dist = {avg_d:.6g} -> alpha = {alpha:.6g}")

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha=alpha
    )
    mesh.compute_vertex_normals()
    logger.info(
        f"Alpha shape produced {len(mesh.vertices):,} vertices and "
        f"{len(mesh.triangles):,} triangles."
    )
    return mesh


def _mesh_planes(
    pcd,
    distance_threshold: float = 0.02,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    min_plane_ratio: float = 0.05,
    max_planes: int = 10,
    bpa_radii=None,
):
    """Plane-based reconstruction for indoor scenes.

    1. Iteratively run RANSAC plane detection.
    2. For each plane, project inliers, compute 2-D convex hull, triangulate.
    3. Collect all plane meshes into one.
    4. Remaining non-plane points are meshed with BPA.
    """
    import open3d as o3d

    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    n_total = len(pts)
    min_points = max(int(min_plane_ratio * n_total), 100)

    plane_meshes: list[o3d.geometry.TriangleMesh] = []
    remaining_mask = np.ones(n_total, dtype=bool)

    logger.info(
        f"Plane detection (distance_threshold={distance_threshold}, "
        f"min_plane_ratio={min_plane_ratio}, max_planes={max_planes})..."
    )

    for plane_idx in range(max_planes):
        if remaining_mask.sum() < min_points:
            break

        remaining_pcd = pcd.select_by_index(np.where(remaining_mask)[0])
        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )

        if len(inliers) < min_points:
            logger.info(
                f"  Plane {plane_idx}: only {len(inliers)} inliers "
                f"(< {min_points}), stopping."
            )
            break

        normal = np.array(plane_model[:3])
        normal = normal / (np.linalg.norm(normal) + 1e-12)

        inlier_indices = np.where(remaining_mask)[0][inliers]
        inlier_pts = pts[inlier_indices]
        inlier_cols = cols[inlier_indices]

        abs_normal = np.abs(normal)
        if abs_normal[0] < abs_normal[1] and abs_normal[0] < abs_normal[2]:
            tangent = np.array([1.0, 0.0, 0.0])
        elif abs_normal[1] < abs_normal[2]:
            tangent = np.array([0.0, 1.0, 0.0])
        else:
            tangent = np.array([0.0, 0.0, 1.0])
        tangent = tangent - np.dot(tangent, normal) * normal
        tangent = tangent / (np.linalg.norm(tangent) + 1e-12)
        bitangent = np.cross(normal, tangent)

        centroid = inlier_pts.mean(axis=0)
        centered = inlier_pts - centroid
        u = np.dot(centered, tangent)
        v = np.dot(centered, bitangent)
        uv = np.column_stack((u, v))

        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(uv)
            hull_vertices_uv = uv[hull.vertices]
            hull_indices = hull.vertices
        except Exception:
            logger.warning(
                f"  Plane {plane_idx}: convex hull failed, using bbox fallback."
            )
            min_u, min_v = uv.min(axis=0)
            max_u, max_v = uv.max(axis=0)
            hull_vertices_uv = np.array(
                [
                    [min_u, min_v],
                    [max_u, min_v],
                    [max_u, max_v],
                    [min_u, max_v],
                ]
            )
            hull_indices = np.arange(4)

        if len(hull_vertices_uv) < 3:
            logger.warning(
                f"  Plane {plane_idx}: only {len(hull_vertices_uv)} hull vertices, skipping."
            )
            remaining_mask[inlier_indices] = False
            continue

        centroid_uv = hull_vertices_uv.mean(axis=0)
        fan_vertices_uv = np.vstack([hull_vertices_uv, centroid_uv])
        n_hull = len(hull_vertices_uv)
        fan_tris = []
        for i in range(n_hull):
            i_next = (i + 1) % n_hull
            fan_tris.append([i, i_next, n_hull])
        fan_tris = np.array(fan_tris)

        fan_vertices_3d = (
            centroid
            + fan_vertices_uv[:, 0:1] * tangent
            + fan_vertices_uv[:, 1:2] * bitangent
        )

        plane_mesh = o3d.geometry.TriangleMesh()
        plane_mesh.vertices = o3d.utility.Vector3dVector(fan_vertices_3d)
        plane_mesh.triangles = o3d.utility.Vector3iVector(fan_tris)
        plane_mesh.compute_vertex_normals()

        hull_pts_3d = (
            centroid
            + hull_vertices_uv[:, 0:1] * tangent
            + hull_vertices_uv[:, 1:2] * bitangent
        )
        centroid_color = inlier_cols.mean(axis=0).reshape(1, -1)
        ordered_colors = np.vstack([inlier_cols[hull_indices], centroid_color])
        plane_mesh.vertex_colors = o3d.utility.Vector3dVector(ordered_colors)

        plane_meshes.append(plane_mesh)
        remaining_mask[inlier_indices] = False

        logger.info(
            f"  Plane {plane_idx}: {len(inliers):,} inliers "
            f"({len(inliers)/n_total*100:.1f}%) | normal=({normal[0]:+.3f}, {normal[1]:+.3f}, {normal[2]:+.3f})"
        )

    n_remaining = int(remaining_mask.sum())
    logger.info(
        f"Plane stage done: {len(plane_meshes)} planes, "
        f"{n_remaining:,} remaining points ({n_remaining/n_total*100:.1f}%)."
    )

    if plane_meshes:
        combined = plane_meshes[0]
        for m in plane_meshes[1:]:
            combined += m
    else:
        combined = o3d.geometry.TriangleMesh()

    if n_remaining > 50:
        logger.info("Meshing remaining non-plane points with BPA...")
        remaining_pcd = pcd.select_by_index(np.where(remaining_mask)[0])
        bpa_mesh = _run_bpa(remaining_pcd, bpa_radius_mult=2.0, bpa_radii_levels=3)
        combined += bpa_mesh
    else:
        logger.info("Too few remaining points for BPA, skipping.")

    return combined


def point_cloud_to_mesh(
    glb_path: str,
    output_path: str,
    algo: str = "poisson",
    # Poisson params
    depth: int = 10,
    scale: float = 1.0,
    linear_fit: bool = True,
    point_weight: float = 15.0,
    density_quantile: float = 0.4,
    smooth_iters: int = 0,
    # BPA params
    bpa_radius_mult: float = 2.0,
    bpa_radii_levels: int = 3,
    # Alpha params
    alpha_mult: float = 2.0,
    # Plane params
    plane_distance_threshold: float = 0.02,
    plane_ransac_n: int = 3,
    plane_num_iterations: int = 1000,
    plane_min_plane_ratio: float = 0.05,
    plane_max_planes: int = 10,
    # Shared params
    outlier_nb: int = 30,
    outlier_std: float = 1.5,
    normal_knn: int = 30,
    orient_normals_camera: tuple[float, float, float] | None = None,
):
    """Read a colored ``.glb`` point cloud and write a colored ``.glb`` mesh.

    Args:
        glb_path: input colored point cloud.
        output_path: where to write the colored mesh ``.glb``.
        algo: ``"poisson"`` (smooth + watertight), ``"bpa"`` (Ball Pivoting;
            maximally rigid), ``"alpha"`` (Alpha Shape; middle ground), or
            ``"planes"`` (RANSAC plane detection + convex hull + BPA fallback).
        depth, scale, linear_fit, point_weight, density_quantile, smooth_iters:
            Poisson params; ignored when algo != "poisson".
        bpa_radius_mult: BPA ball radius as a multiple of the cloud's average
            nearest-neighbor distance. Smaller = more holes, larger = bridges.
        bpa_radii_levels: number of geometric radii levels (each 2x previous).
        alpha_mult: Alpha shape radius as a multiple of average nearest-neighbor
            distance. Smaller = tighter fit (more holes), larger = fills gaps.
            Ignored when algo != "alpha".
        plane_distance_threshold: RANSAC plane distance threshold (planes).
        plane_ransac_n: RANSAC sample size (planes).
        plane_num_iterations: RANSAC iterations per plane (planes).
        plane_min_plane_ratio: minimum fraction of total points for a plane
            to be accepted (planes).
        plane_max_planes: max planes to detect before stopping (planes).
        outlier_nb / outlier_std: ``remove_statistical_outlier`` params.
        normal_knn: K-nearest neighbors used in normal estimation.
        orient_normals_camera: optional (x, y, z) camera location to orient
            normals toward. None (default) uses
            ``orient_normals_consistent_tangent_plane`` which is the correct
            choice for orbit captures.
    """
    if algo not in ("poisson", "bpa", "alpha", "planes"):
        raise ValueError(f"algo must be 'poisson', 'bpa', 'alpha', or 'planes', got {algo!r}")

    pts, cols = _load_pointcloud(glb_path)

    import open3d as o3d
    pcd = _build_pcd(
        pts, cols, outlier_nb, outlier_std, normal_knn, orient_normals_camera
    )

    if algo == "poisson":
        mesh = _run_poisson(
            pcd, depth, scale, linear_fit, point_weight, density_quantile
        )
    elif algo == "bpa":
        mesh = _run_bpa(pcd, bpa_radius_mult, bpa_radii_levels)
    elif algo == "alpha":
        mesh = _run_alpha(pcd, alpha_mult)
    else:  # algo == "planes"
        mesh = _mesh_planes(
            pcd,
            distance_threshold=plane_distance_threshold,
            ransac_n=plane_ransac_n,
            num_iterations=plane_num_iterations,
            min_plane_ratio=plane_min_plane_ratio,
            max_planes=plane_max_planes,
        )

    logger.info(
        "Transferring colors from point cloud to mesh (vectorized cKDTree)..."
    )
    from scipy.spatial import cKDTree

    mesh_vertices = np.asarray(mesh.vertices)
    pcd_colors = np.asarray(pcd.colors)
    pcd_vertices = np.asarray(pcd.points)
    kdtree = cKDTree(pcd_vertices)
    _distances, indices = kdtree.query(mesh_vertices, k=1, workers=-1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(pcd_colors[indices])

    if smooth_iters > 0 and algo == "poisson":
        logger.info(f"Smoothing mesh (Laplacian, {smooth_iters} iter)...")
        mesh = mesh.filter_smooth_simple(number_of_iterations=smooth_iters)
    elif smooth_iters > 0 and algo in ("bpa", "alpha", "planes"):
        logger.warning(
            f"smooth_iters > 0 was passed with algo={algo}; ignoring "
            "(these algos are designed for rigid, unsmoothed output)."
        )
    mesh.compute_vertex_normals()

    logger.info(f"Exporting colored mesh to {output_path}...")
    # Flip Y and Z so the mesh matches the glTF convention of the point cloud.
    # (Open3D's native GLB writer preserves raw coordinates; the cloud was
    # already exported viewer-correct by DUSt3R, so we align the mesh here.)
    mesh_vertices = np.asarray(mesh.vertices)
    mesh.vertices = o3d.utility.Vector3dVector(
        mesh_vertices * np.array([1.0, -1.0, -1.0])
    )

    o3d.io.write_triangle_mesh(
        output_path,
        mesh,
        write_triangle_uvs=True,
        print_progress=False,
    )
    logger.info(
        f"Mesh complete ({algo}): {len(mesh.vertices):,} vertices, "
        f"{len(mesh.triangles):,} faces -> {output_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a colored .glb point cloud into a colored .glb mesh. "
        "Choose between Poisson (smooth + watertight, strict-tuned) and BPA "
        "(maximally rigid; mesh vertices exactly equal input points)."
    )
    parser.add_argument("--input", required=True, help="Input .glb point cloud")
    parser.add_argument("--output", required=True, help="Output .glb mesh")
    parser.add_argument(
        "--algo", choices=("poisson", "bpa", "alpha", "planes"), default="poisson",
        help="Meshing algorithm. poisson=smooth watertight, bpa=maximally rigid, "
             "alpha=middle ground, planes=RANSAC planes + convex hull + BPA fallback.",
    )
    # Poisson knobs.
    parser.add_argument(
        "--depth", type=int, default=10,
        help="[poisson] Octree depth. 10=recommended, 11=needs ~16 GB RAM.",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="[poisson] Bounding-box inflation factor (1.0=tight).",
    )
    parser.add_argument(
        "--no_linear_fit", action="store_true",
        help="[poisson] Use Catmull-Clark iso-extraction (smoother, default in "
             "Open3D) instead of linear (sharper, our default).",
    )
    parser.add_argument(
        "--point_weight", type=float, default=15.0,
        help="[poisson] Iso-surface stickiness to input points. Default 4 in "
             "Open3D, 15 here for stricter fit. 20+ = very rigid.",
    )
    parser.add_argument(
        "--density_quantile", type=float, default=0.4,
        help="[poisson] Drop mesh vertices below this density quantile. "
             "0.05=lenient, 0.4=ours, 0.5=aggressive.",
    )
    parser.add_argument(
        "--smooth_iters", type=int, default=0,
        help="[poisson] Laplacian smoothing iterations after Poisson. "
             "0=none (recommended for strict fit), 1=Open3D example default.",
    )
    # BPA knobs.
    parser.add_argument(
        "--bpa_radius_mult", type=float, default=2.0,
        help="[bpa] Ball radius as multiple of avg nearest-neighbor distance. "
             "Smaller=more holes, larger=bridges across gaps (and across true "
             "depth discontinuities, which is usually undesirable).",
    )
    parser.add_argument(
        "--bpa_radii_levels", type=int, default=3,
        help="[bpa] # of geometric radii levels (each 2x previous). 3=default.",
    )
    # Alpha knobs.
    parser.add_argument(
        "--alpha_mult", type=float, default=2.0,
        help="[alpha] Alpha radius as multiple of avg nearest-neighbor distance. "
             "Smaller=tighter fit (more holes), larger=fills gaps. 2.0=default.",
    )
    # Plane knobs.
    parser.add_argument(
        "--plane_distance_threshold", type=float, default=0.02,
        help="[planes] RANSAC plane distance threshold. Smaller=stricter fit.",
    )
    parser.add_argument(
        "--plane_min_plane_ratio", type=float, default=0.05,
        help="[planes] Minimum fraction of total points for a valid plane.",
    )
    parser.add_argument(
        "--plane_max_planes", type=int, default=10,
        help="[planes] Max planes to detect.",
    )
    # Shared knobs.
    parser.add_argument(
        "--outlier_nb", type=int, default=30,
        help="Statistical outlier removal: number of neighbors.",
    )
    parser.add_argument(
        "--outlier_std", type=float, default=1.5,
        help="Statistical outlier removal: std deviation cutoff (lower=stricter).",
    )
    parser.add_argument(
        "--normal_knn", type=int, default=30,
        help="K-nearest neighbors used in normal estimation (scale-invariant).",
    )
    parser.add_argument(
        "--orient_normals_camera", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help="Optional camera location to orient normals toward. Default uses "
             "consistent-tangent-plane propagation (correct for orbits).",
    )
    args = parser.parse_args()

    point_cloud_to_mesh(
        args.input,
        args.output,
        algo=args.algo,
        depth=args.depth,
        scale=args.scale,
        linear_fit=not args.no_linear_fit,
        point_weight=args.point_weight,
        density_quantile=args.density_quantile,
        smooth_iters=args.smooth_iters,
        bpa_radius_mult=args.bpa_radius_mult,
        bpa_radii_levels=args.bpa_radii_levels,
        alpha_mult=args.alpha_mult,
        plane_distance_threshold=args.plane_distance_threshold,
        plane_min_plane_ratio=args.plane_min_plane_ratio,
        plane_max_planes=args.plane_max_planes,
        outlier_nb=args.outlier_nb,
        outlier_std=args.outlier_std,
        normal_knn=args.normal_knn,
        orient_normals_camera=tuple(args.orient_normals_camera)
        if args.orient_normals_camera is not None
        else None,
    )
