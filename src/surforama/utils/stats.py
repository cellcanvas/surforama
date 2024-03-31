import numpy as np
import potpourri3d as pp3d
import pyvista as pv
from pygeodesic import geodesic

from surforama.constants import NAPARI_UP_0, NAPARI_UP_1, NAPARI_UP_2


def find_closest_vertex(vertices, point):
    """Find the index of the closest vertex to a given point."""
    distances = np.linalg.norm(vertices - point, axis=1)
    return np.argmin(distances)


def compute_geod_distance_matrix(
    verts, faces, point_coordinates, method="exact"
):
    """
    Compute the geodesic distance matrix between a set of points on a mesh.

    Has two methods for computing the geodesic distances:
    - "exact": Uses the exact geodesic algorithm from the pygeodesic library.
        (takes forever for large meshes and many points)
    - "fast": Uses the mesh heat method distance solver from the potpourri3d library.
        (faster but less accurate -- but still good enough for most applications;
        differences are in the area of ~[-0.2, 0.2] vx for the distances between points on the same mesh)

    Parameters
    ----------
    verts : np.ndarray
        The vertices of the mesh.
    faces : np.ndarray
        The faces of the mesh.
    point_coordinates : np.ndarray
        The coordinates of the points for which the geodesic distances should be computed.
    method : str
        The method to use for computing the geodesic distances. Can be either "exact" or "fast".

    Returns
    -------
    np.ndarray
        The geodesic distance matrix between the points.

    """

    if method == "exact":
        geoalg = geodesic.PyGeodesicAlgorithmExact(verts, faces)
    elif method == "fast":
        solver = pp3d.MeshHeatMethodDistanceSolver(V=verts, F=faces)

    point_idcs = [
        find_closest_vertex(verts, point) for point in point_coordinates
    ]

    distance_matrix = np.zeros((len(point_idcs), len(point_idcs))).astype(
        np.float32
    )
    for i, point_idx in enumerate(point_idcs):
        if method == "exact":
            distances, _ = geoalg.geodesicDistances(
                np.array([point_idx]), np.arange(len(verts))
            )
        elif method == "fast":
            distances = solver.compute_distance(point_idx)

        distances[point_idx] = 1e5
        distances = distances[point_idcs]
        distance_matrix[i] = distances

    return distance_matrix


def compute_surface_occupancy(
    verts, faces, point_coordinates, only_front=True
):
    """
    Compute the surface occupancy of a set of points on a mesh.

    The surface occupancy is defined as the number of points divided by the surface area
    of the mesh that is covered by the points.

    Parameters
    ----------
    verts : np.ndarray
        The vertices of the mesh.
    faces : np.ndarray
        The faces of the mesh.
    point_coordinates : np.ndarray
        The coordinates of the points for which the surface occupancy should be computed.
    only_front : bool
        If True, only the front side of the mesh is considered for the surface area computation.
        This means that the surface area is divided by 2.

    Returns
    -------
    float
        The surface occupancy of the points on the mesh.
    """
    pv_faces = np.hstack([np.ones((faces.shape[0], 1)) * 3, faces])
    pv_faces = pv_faces.flatten().astype(np.int32)
    pv_mesh = pv.PolyData(verts, pv_faces)
    area = pv_mesh.area
    if only_front:
        area /= 2.0  # divide by 2 to only take the front side
    return len(point_coordinates) / area


def orientations_of_knn_inplane(
    distance_matrix, feature_table, k=3, c2_symmetry=False
):
    """
    Compute the angular differences between the up vectors of a point and its k nearest neighbors.

    The angular differences are computed in degrees and are in the range [0, 180] if c2_symmetry is False
    and in the range [0, 90] if c2_symmetry is True.

    Parameters
    ----------
    distance_matrix : np.ndarray
        The distance matrix between the points.
    feature_table : pd.DataFrame
        The feature table containing the up vectors of the points.
    k : int
        The number of nearest neighbors to consider.
    c2_symmetry : bool
        Whether to consider the c2 symmetry of the up vectors.
        (implies C2 symmetry of the respective protein)

    Returns
    -------
    list
        A list of angular differences for each point. (in degrees)
    """
    up_vectors = feature_table[
        [NAPARI_UP_0, NAPARI_UP_1, NAPARI_UP_2]
    ].to_numpy()
    nn_orientations = []
    for i in range(distance_matrix.shape[0]):
        start_vector = up_vectors[i]
        nn_idx = np.argsort(distance_matrix[i])[:k]
        cosine_similarities = np.degrees(
            np.arccos(np.dot(start_vector, up_vectors[nn_idx].T))
        )
        if c2_symmetry:
            cosine_similarities = np.minimum(
                cosine_similarities, 180 - cosine_similarities
            )
        nn_orientations.append(cosine_similarities)
    return np.array(nn_orientations)
