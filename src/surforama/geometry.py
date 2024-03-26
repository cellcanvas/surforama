import numpy as np
from scipy.spatial.transform import Rotation as R


def rotate_around_vector(
    rotate_around: np.ndarray, to_rotate: np.ndarray, angle: float
) -> np.ndarray:
    """Rotate a vector around another vector a specified angle.

    Parameters
    ----------
    rotate_around : np.ndarray
        The vector to rotate around.
    to_rotate : np.ndarray
        The vector to rotate.
    angle : float
        The angle to perform the rotation in radians.
        Positive angles will rotate counter-clockwise.

    Returns
    -------
    np.ndarray
        The rotated vector.
    """
    # ensure unit vectors
    rotate_around = rotate_around.astype(float)
    rotate_around /= np.linalg.norm(rotate_around)

    rotation_vector = angle * rotate_around
    rotation_object = R.from_rotvec(rotation_vector)

    return rotation_object.apply(to_rotate)
