import numpy as np

from surforama.utils.geometry import rotate_around_vector


def test_rotate_around_axis_single_vector():
    rotate_around = np.array([1, 0, 0])
    to_rotate = np.array([0, 1, 0])

    rotation_amount = np.pi / 2
    rotated_vector = rotate_around_vector(
        rotate_around=rotate_around, to_rotate=to_rotate, angle=rotation_amount
    )

    assert np.allclose(rotated_vector, np.array([0, 0, 1]))


def test_rotate_around_axis_multiple_vectors():
    rotate_around = np.array([[1, 0, 0], [0, 1, 0]])
    to_rotate = np.array([[0, 1, 0], [0, 0, 1]])

    rotation_amount = np.pi / 2
    rotated_vector = rotate_around_vector(
        rotate_around=rotate_around, to_rotate=to_rotate, angle=rotation_amount
    )

    assert np.allclose(rotated_vector, np.array([[0, 0, 1], [1, 0, 0]]))


def test_rotate_around_axis_multiple_vectors_multiple_angles():
    rotate_around = np.array([[1, 0, 0], [0, 1, 0]])
    to_rotate = np.array([[0, 1, 0], [0, 0, 1]])

    rotation_amount = np.array([np.pi / 2, np.pi])
    rotated_vector = rotate_around_vector(
        rotate_around=rotate_around, to_rotate=to_rotate, angle=rotation_amount
    )

    assert np.allclose(rotated_vector, np.array([[0, 0, 1], [0, 0, -1]]))
