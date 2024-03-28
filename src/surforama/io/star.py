from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import starfile
from napari.layers import Points
from scipy.spatial.transform import Rotation as R

from surforama.constants import (
    NAPARI_NORMAL_0,
    NAPARI_NORMAL_1,
    NAPARI_NORMAL_2,
    NAPARI_UP_0,
    NAPARI_UP_1,
    NAPARI_UP_2,
    ROTATION,
    STAR_ROTATION_0,
    STAR_ROTATION_1,
    STAR_ROTATION_2,
    STAR_X_COLUMN_NAME,
    STAR_Y_COLUMN_NAME,
    STAR_Z_COLUMN_NAME,
)
from surforama.utils.geometry import rotate_around_vector


def load_points_from_star_table(star_table: pd.DataFrame) -> np.ndarray:
    """Get point coordinates from a Relion-formatted star table.

    Currently this does not account for shifts.

    Parameters
    ----------
    star_table : pd.DataFrame
        The table from which to extract the point coordinates
    """
    return star_table[
        [STAR_Z_COLUMN_NAME, STAR_Y_COLUMN_NAME, STAR_X_COLUMN_NAME]
    ].to_numpy()


def load_orientations_from_star_table(star_table: pd.DataFrame):
    """Get orientations from a Relion-formatted star table."""
    eulers = star_table[
        ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]
    ].to_numpy()
    return R.from_euler(seq="ZYZ", angles=eulers, degrees=True).inv()


def load_points_layer_data_from_star_file(
    file_path: str,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load an oriented Points layer from a Relion-formatted star file"""
    star_table = starfile.read(file_path)
    point_coordinates = load_points_from_star_table(star_table)
    orientations = load_orientations_from_star_table(star_table)

    # get the normal vectors
    initial_normal_vector = np.array([0, 0, 1])
    normal_vectors = orientations.apply(initial_normal_vector)[:, ::-1]

    # get the up vectors
    initial_up_vector = np.array([0, 1, 0])
    up_vectors = orientations.apply(initial_up_vector)[:, ::-1]

    # initialize the rotations
    n_points = point_coordinates.shape[0]
    rotations = np.zeros((n_points,))

    feature_table = pd.DataFrame(
        {
            NAPARI_NORMAL_0: normal_vectors[:, 0],
            NAPARI_NORMAL_1: normal_vectors[:, 1],
            NAPARI_NORMAL_2: normal_vectors[:, 2],
            NAPARI_UP_0: up_vectors[:, 0],
            NAPARI_UP_1: up_vectors[:, 1],
            NAPARI_UP_2: up_vectors[:, 2],
            ROTATION: rotations,
        }
    )

    return point_coordinates, feature_table


def oriented_points_to_star_table(points_layer: Points):
    """points with orientations to a star-formatted DataFrame."""
    # get the point coordinates
    napari_coordinates = points_layer.data
    relion_coordinates = napari_coordinates[:, ::-1]

    # get the point orientations
    features_table = points_layer.features
    normal_vectors = features_table[
        [NAPARI_NORMAL_0, NAPARI_NORMAL_1, NAPARI_NORMAL_2]
    ].to_numpy()
    up_vectors = features_table[
        [NAPARI_UP_0, NAPARI_UP_1, NAPARI_UP_2]
    ].to_numpy()
    rotations = features_table[ROTATION].to_numpy()
    rotated_up_vectors = rotate_around_vector(
        rotate_around=normal_vectors, to_rotate=up_vectors, angle=rotations
    )
    third_basis = np.cross(normal_vectors, rotated_up_vectors)
    n_points = napari_coordinates.shape[0]
    particle_orientations = np.zeros((n_points, 3, 3))
    particle_orientations[:, :, 0] = third_basis[:, ::-1]
    particle_orientations[:, :, 1] = rotated_up_vectors[:, ::-1]
    particle_orientations[:, :, 2] = normal_vectors[:, ::-1]
    euler_angles = (
        R.from_matrix(particle_orientations)
        .inv()
        .as_euler(seq="ZYZ", degrees=True)
    )

    return pd.DataFrame(
        {
            STAR_X_COLUMN_NAME: relion_coordinates[:, 0],
            STAR_Y_COLUMN_NAME: relion_coordinates[:, 1],
            STAR_Z_COLUMN_NAME: relion_coordinates[:, 2],
            STAR_ROTATION_0: euler_angles[:, 0],
            STAR_ROTATION_1: euler_angles[:, 1],
            STAR_ROTATION_2: euler_angles[:, 2],
        }
    )


def oriented_points_to_star_file(
    points_layer: Points, output_path: Union[str, Path]
):
    """points with orientations to a star file"""
    star_table = oriented_points_to_star_table(points_layer)
    starfile.write(star_table, output_path)
