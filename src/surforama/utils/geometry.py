from typing import Union

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R


def rotate_around_vector(
    rotate_around: np.ndarray,
    to_rotate: np.ndarray,
    angle: Union[float, np.ndarray],
) -> np.ndarray:
    """Rotate a vector around another vector a specified angle.

    Parameters
    ----------
    rotate_around : np.ndarray
        The vector to rotate around.
    to_rotate : np.ndarray
        The vector to rotate.
    angle : Union[float, np.ndarray]
        The angle to perform the rotation in radians.
        Positive angles will rotate counter-clockwise.
        If provided as an array, each element should be indexed-matched
        with the vectors.

    Returns
    -------
    np.ndarray
        The rotated vector.
    """
    # ensure unit vectors
    rotate_around = rotate_around.astype(float)
    if rotate_around.ndim == 1:
        rotate_around /= np.linalg.norm(rotate_around)
    elif rotate_around.ndim == 2:
        # handle the 2D case
        rotate_around /= np.linalg.norm(rotate_around, axis=1)[..., np.newaxis]
        angle = np.asarray(angle)[..., np.newaxis]
    else:
        raise ValueError(
            "rotate_around must be a single vector as a 1D array or a 2D array of multiple vectors."
        )

    rotation_vector = angle * rotate_around
    rotation_object = R.from_rotvec(rotation_vector)

    return rotation_object.apply(to_rotate)


def find_closest_triangle(mesh: trimesh.Trimesh, point: np.ndarray) -> int:
    """
    Find the index of the triangle in a mesh that is closest to a specified point.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh in which to find the closest triangle.
    point : np.ndarray
        A 3D point (as a NumPy array) for which the closest triangle is to be found.

    Returns
    -------
    int
        The index of the triangle that is closest to the given point.

    Notes
    -----
    This function calculates the geometric center of each triangle and finds the one
    closest to the specified point using Euclidean distance.
    """
    triangle_centers = np.mean(mesh.vertices[mesh.faces], axis=1)
    distances = np.linalg.norm(triangle_centers - point, axis=1)
    return np.argmin(distances)


def find_closest_normal(
    mesh: trimesh.Trimesh, point: np.ndarray
) -> np.ndarray:
    """
    Find the normal of the closest triangle in a mesh to a specified point.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh from which to find the closest triangle normal.
    point : np.ndarray
        A 3D point (as a NumPy array) for which the closest triangle normal is to be found.

    Returns
    -------
    np.ndarray
        The normal vector of the closest triangle to the given point.

    Notes
    -----
    This function uses `find_closest_triangle` to determine the closest triangle and then
    retrieves the normal vector associated with that triangle from the mesh's `face_normals`.
    """
    triangle_index = find_closest_triangle(mesh, point)
    face_normals = mesh.face_normals
    return face_normals[triangle_index]
