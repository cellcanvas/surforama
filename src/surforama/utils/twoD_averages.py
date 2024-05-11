import numpy as np
import trimesh
from scipy.ndimage import map_coordinates

from surforama.utils.geometry import find_closest_normal


def define_rotation_kernel(shape: tuple) -> np.ndarray:
    """
    Define a set of rotation kernels based on a specified shape.

    Parameters
    ----------
    shape : tuple
        The shape (dimensions) for which the kernel is to be defined.

    Returns
    -------
    np.ndarray
        A 3D array containing rotation kernels for each radius.
    """
    img_center = np.round(0.5 * np.asarray(shape)) - 1
    rad_voxels = min(
        int(shape[0] - img_center[0]), int(shape[1] - img_center[1])
    )  # number of radius rings

    X, Y = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), indexing="ij"
    )  # X, Y, meshgrid for x and y coordinates
    kernels = np.zeros(
        (rad_voxels, shape[0], shape[1]), dtype=float
    )  # kernels, rotation kernels

    distances_to_center = np.sqrt(
        (X - img_center[0] - 0.5) ** 2 + (Y - img_center[1] - 0.5) ** 2
    )
    for i in range(rad_voxels):
        rad_ring = (
            np.abs(distances_to_center - i) - 2
        )  # ring of radius i with thickness 2
        rad_ring[rad_ring > 0] = (
            0  # set everything further than two voxels from the ring to zero
        )
        kernels[i, :, :] = -0.5 * rad_ring

    return kernels


def avg_vol_2D(vol: np.ndarray, mirror: bool = False) -> np.ndarray:
    """
    Rotationally average an input volume to create a 2D image.

    Parameters
    ----------
    vol : np.ndarray
        A 3D numpy array representing the volume.
    mirror : bool, optional
        Flag to mirror the averaged results horizontally.

    Returns
    -------
    np.ndarray
        A 2D image representing the averaged volume.
    """
    shape = vol.shape
    kernels = define_rotation_kernel(shape)

    # Average
    avg = np.zeros(
        (min(vol.shape[0], vol.shape[1]), kernels.shape[0]), dtype=float
    )

    for i in range(vol.shape[2]):
        hold = vol[:, :, i]
        for j, kernel in enumerate(kernels):
            avg[i, j] = (hold * kernel).sum() / (kernel.sum() + 1e-6)

    if mirror:
        avg = np.concatenate((avg[:, -1:0:-1], avg), axis=1)
    return avg


def extract_normal_volume(
    point: np.ndarray, normal: np.ndarray, tomogram: np.ndarray, shape: tuple
) -> np.ndarray:
    """
    Extract a volume from a tomogram that is aligned with a point and a normal vector.

    Parameters
    ----------
    point : np.ndarray
        A 3D point in the tomogram.
    normal : np.ndarray
        The normal vector.
    tomogram : np.ndarray
        A 3D numpy array of the tomogram.
    shape : tuple
        The desired shape of the extracted volume.

    Returns
    -------
    np.ndarray
        A 3D numpy array representing the extracted volume aligned with the normal vector.
    """
    # Normalize the normal vector
    z_axis = normal / np.linalg.norm(normal)

    # Arbitrary choice for X-axis (just make sure it's not parallel to Z)
    x_axis = (
        np.array([1, 0, 0])
        if z_axis[0] == 0 or z_axis[1] == 0
        else np.array([0, 0, 1])
    )
    x_axis = (
        x_axis - np.dot(x_axis, z_axis) * z_axis
    )  # Remove the component parallel to Z
    x_axis /= np.linalg.norm(x_axis)  # Normalize
    # Y-axis to complete the right-handed system
    y_axis = np.cross(z_axis, x_axis)

    # Create rotation matrix from the original axes to the new axes
    rotation_matrix = np.array([x_axis, y_axis, z_axis])

    # Define the local coordinates around the point
    local_x = np.linspace(-shape[0] / 2, shape[0] / 2, shape[0])
    local_y = np.linspace(-shape[1] / 2, shape[1] / 2, shape[1])
    local_z = np.linspace(-shape[2] / 2, shape[2] / 2, shape[2])
    local_grid = np.array(
        np.meshgrid(local_x, local_y, local_z, indexing="ij")
    )

    # Flatten and rotate the grid, then add the point
    local_grid_flat = local_grid.reshape(3, -1)
    global_grid_flat = np.dot(rotation_matrix.T, local_grid_flat).T + point

    # Use scipy's map_coordinates to extract the aligned volume
    extracted_volume = map_coordinates(
        tomogram, global_grid_flat.T, order=1, mode="nearest"
    ).reshape(shape)

    return extracted_volume


def create_2D_averages(
    positions: list,
    mesh: trimesh.Trimesh,
    tomogram: np.ndarray,
    shape: tuple = (20, 20, 20),
    mirror: bool = True,
) -> np.ndarray:
    """
    Create 2D averages from a list of positions using a given mesh and tomogram.

    Parameters
    ----------
    positions : list
        A list of 3D points.
    mesh :
        A mesh object used to find normals corresponding to the given positions.
    tomogram : np.ndarray
        A 3D numpy array containing the tomogram data.
    mirror : bool, optional
        Flag to mirror the results horizontally for each 2D average.

    Returns
    -------
    np.ndarray
        An array of 2D averages calculated for each position.

    """
    averages = []
    for position in positions:
        normal = find_closest_normal(mesh, position)
        volume = extract_normal_volume(position, normal, tomogram, shape)
        avg = avg_vol_2D(volume, mirror=mirror)
        averages.append(avg)
    return np.array(averages)


def normalize_averages(avgs: np.ndarray) -> np.ndarray:
    """
    Normalize the average images.

    This is done by subtracting the mean and dividing by the standard deviation.
    Mean and standard deviation are calculated across all the averages.

    Parameters
    ----------
    avgs : np.ndarray
        The array of averages to be normalized.

    Returns
    -------
    np.ndarray
        The normalized averages.
    """
    mean = np.mean(avgs)
    std = np.std(avgs)
    avgs = (avgs - mean) / std
    return avgs
