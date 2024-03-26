import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from surforama.constants import (
    STAR_X_COLUMN_NAME,
    STAR_Y_COLUMN_NAME,
    STAR_Z_COLUMN_NAME,
)


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


def points_to_star_table():
    """napari point coordinates to a star-formatted table.

    This uses the Relion column conventions.
    """


def rotation_to_star_table():
    """napari rotation coordinate system to a star-formatted table.

    This uses the Relion column conventions.
    """


def oriented_points_to_star_table():
    """points with orientations to a star-formatted table."""
