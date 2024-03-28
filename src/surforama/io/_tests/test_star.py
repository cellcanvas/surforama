import numpy as np
import pandas as pd
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
from surforama.io.star import (
    load_points_layer_data_from_star_file,
    oriented_points_to_star_file,
)
from surforama.utils.geometry import rotate_around_vector


def test_star_file_round_trip(tmp_path):
    """Test saving and loading a star file is lossless."""
    # make a points layer with an oriented points feature table
    point_coordinates = np.array([[10, 10, 10], [20, 20, 20]])
    normal_vectors = np.array([[1, 0, 0], [1, 0, 0]])
    up_vectors = np.array([[0, 0, 1], [0, 1, 0]])
    rotations = np.array([0, np.pi / 2])
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
    points_layer = Points(point_coordinates, features=feature_table)

    # save the feature table to a star file
    star_file_path = tmp_path / "test.star"
    oriented_points_to_star_file(points_layer, output_path=star_file_path)

    # load the star file
    returned_points_coordinates, returned_features_table = (
        load_points_layer_data_from_star_file(star_file_path)
    )

    # compare the point coordinates
    np.testing.assert_allclose(returned_points_coordinates, point_coordinates)

    # compare the normal vectors
    returned_normal_vectors = returned_features_table[
        [NAPARI_NORMAL_0, NAPARI_NORMAL_1, NAPARI_NORMAL_2]
    ]
    np.testing.assert_allclose(returned_normal_vectors, normal_vectors)

    # compare the up vectors
    expected_up_vectors = rotate_around_vector(
        rotate_around=normal_vectors, to_rotate=up_vectors, angle=rotations
    )
    returned_up_vectors = returned_features_table[
        [NAPARI_UP_0, NAPARI_UP_1, NAPARI_UP_2]
    ].to_numpy()
    np.testing.assert_allclose(
        returned_up_vectors, expected_up_vectors, atol=1e-15
    )
