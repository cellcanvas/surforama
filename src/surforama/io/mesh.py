import numpy as np
import pyacvd
import pyvista as pv
import trimesh
from skimage.measure import marching_cubes


def read_obj_file(file_path):
    mesh = trimesh.load(file_path, file_type="obj", process=True)

    # Subdivide
    # verts, faces = trimesh.remesh.subdivide_to_size(
    # mesh.vertices, mesh.faces, 1
    # )

    # Subdivide can introduce holes
    # mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    # trimesh.repair.fill_holes(mesh)

    verts = mesh.vertices
    faces = mesh.faces

    # trimesh swaps coordinate axes
    verts = verts[:, [2, 1, 0]]

    values = np.ones((len(verts),))

    return verts, faces, values


def convert_mask_to_mesh(
    mask: np.ndarray,
    barycentric_area: float = 1.0,
    smoothing: int = 10,
):
    """
    Convert a binary mask to a mesh.

    Parameters
    ----------
    mask : np.ndarray
        A binary mask.
    barycentric_area : float, optional
        The target barycentric area of each vertex in the mesh,
        by default 1.0
    smoothing : int, optional
        Number of iterations for Laplacian mesh smoothing,
        by default 10
    """
    verts, faces, _, _ = marching_cubes(
        volume=mask,
        level=0.5,
        step_size=1,
        method="lewiner",
    )

    # Prepend 3 for pyvista format
    faces = np.concatenate(
        (np.ones((faces.shape[0], 1), dtype=int) * 3, faces), axis=1
    )

    # Create a mesh
    surf = pv.PolyData(verts, faces)
    surf = surf.smooth(n_iter=smoothing)

    # remesh to desired point size
    cluster_points = int(surf.area / barycentric_area)
    clus = pyacvd.Clustering(surf)
    clus.subdivide(3)
    clus.cluster(cluster_points)
    remesh = clus.create_mesh()

    verts = remesh.points
    faces = remesh.faces.reshape(-1, 4)[:, 1:]
    values = np.ones((len(verts),))

    # switch face order to have outward normals
    faces = faces[:, [0, 2, 1]]

    return verts, faces, values
