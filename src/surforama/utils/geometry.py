from typing import Union

import numpy as np
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
