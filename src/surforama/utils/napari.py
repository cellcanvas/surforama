from typing import Tuple

import numpy as np
from napari.layers import Points

from surforama.constants import (
    NAPARI_NORMAL_0,
    NAPARI_NORMAL_1,
    NAPARI_NORMAL_2,
    NAPARI_UP_0,
    NAPARI_UP_1,
    NAPARI_UP_2,
    ROTATION,
)
from surforama.utils.geometry import rotate_around_vector


def vectors_data_from_points_features(
    points_layer: Points,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the vectors data for normal and up vectors from an oriented points layer.

    Parameters
    ----------
    points_layer : Points
        The oriented points layer from which to get the vectors data.

    Returns
    -------
    normals_data : np.ndarray
        The Vectors layer data for the normal vectors
    up_data : np.ndarray
        The Vectors layer data for the up vectors
    """

    # get the point data
    point_coordinates = points_layer.data

    feature_table = points_layer.features

    # get the vectors
    normal_vectors = feature_table[
        [NAPARI_NORMAL_0, NAPARI_NORMAL_1, NAPARI_NORMAL_2]
    ].to_numpy()
    up_vectors = feature_table[
        [NAPARI_UP_0, NAPARI_UP_1, NAPARI_UP_2]
    ].to_numpy()
    rotations = feature_table[ROTATION].to_numpy()
    rotated_up_vectors = rotate_around_vector(
        rotate_around=normal_vectors, to_rotate=up_vectors, angle=rotations
    )

    # make the vector data
    normal_vector_data = np.zeros((1, 2, 3))
    normal_vector_data[:, 0, :] = point_coordinates
    normal_vector_data[:, 1, :] = normal_vectors

    up_vector_data = np.zeros((1, 2, 3))
    up_vector_data[:, 0, :] = point_coordinates
    up_vector_data[:, 1, :] = rotated_up_vectors

    return normal_vector_data, up_vector_data
