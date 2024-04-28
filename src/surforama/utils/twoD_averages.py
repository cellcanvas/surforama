import numpy as np
import trimesh
from scipy.ndimage import map_coordinates
from scipy.signal import correlate2d

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


def noshift_crosscorrelation(img1, img2, mask):
    """This is also the cross-correlation used in PySeg:
    https://github.com/anmartinezs/pyseg_system/blob/925f04b3415bff45c4bdd957d17e9a8cabc2fb8f/code/pyseg/sub/klass.py#L111
    """
    corr = np.sum(img1 * img2) / np.sum(mask)
    return corr


def extract_masked_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Extract a sub-image from the given volume using a mask.

    Parameters
    ----------
    volume : np.ndarray
        The full volume from which to extract.
    mask : np.ndarray
        The mask indicating the sub-volume to extract.

    Returns
    -------
    np.ndarray
        The extracted sub-volume defined by the mask.
    """
    rows, cols = np.where(mask == 1)
    return image[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def build_cc_matrix(avgs: np.ndarray, masks: list) -> np.ndarray:
    """
    Build a cross-correlation matrix for a set of average images using corresponding masks.

    Parameters
    ----------
    avgs : np.ndarray
        Array of average images.
    masks : list
        List of masks corresponding to each image in `avgs`.

    Returns
    -------
    np.ndarray
        A matrix of cross-correlation values between all pairs of images.
    """
    distance_matrix = np.ones((avgs.shape[0], avgs.shape[0])) * 1e6
    avgs = normalize_averages(avgs)
    for k, (img1, mask1) in enumerate(zip(avgs, masks)):
        img1_backup = img1.copy()
        for i, (img2, mask2) in enumerate(zip(avgs, masks)):
            if distance_matrix[k, i] < 1e6:  # already computed
                continue
            mask = mask1 * mask2
            # Extract array of mask shape using the mask indices
            img1 = extract_masked_image(img1_backup.copy(), mask)
            img2 = extract_masked_image(img2.copy(), mask)

            corr = correlate2d(img1, img2, mode="full").max()

            distance_matrix[k, i] = corr
            distance_matrix[i, k] = corr
    return distance_matrix


def cluster_averages(
    avgs: np.ndarray, masks: list = None, method: str = "affinity_propagation"
) -> np.ndarray:
    """
    Cluster averages using a specified method.

    Parameters
    ----------
    avgs : np.ndarray
        Array of average images.
    masks : list, optional
        List of masks corresponding to each average image.
    method : str, optional
        Clustering method to use ('affinity_propagation' or 'spectral_clustering').

    Returns
    -------
    np.ndarray
        Array of cluster labels for each average.

    Raises
    ------
    ValueError
        If the specified method is not implemented.
    """
    if masks is None:
        masks = [np.ones_like(avg) for avg in avgs]
    distance_matrix = build_cc_matrix(avgs, masks)
    # distance_matrix = np.max(distance_matrix) - distance_matrix # convert to distance matrix
    if method == "affinity_propagation":
        from sklearn.cluster import AffinityPropagation

        clustering = AffinityPropagation(affinity="precomputed").fit(
            distance_matrix
        )
    elif method == "spectral_clustering":
        from sklearn.cluster import SpectralClustering

        clustering = SpectralClustering(affinity="precomputed").fit(
            distance_matrix
        )
    else:
        raise ValueError("Method not implemented")
    return clustering.labels_


def get_2D_mask(
    distance_top: int, distance_bottom: int, distance_side: int, shape: tuple
) -> np.ndarray:
    """
    Generate a 2D mask with specified borders masked on each side of an image.

    Parameters
    ----------
    distance_top : int
        The width of the top border to be masked.
    distance_bottom : int
        The width of the bottom border to be masked.
    distance_side : int
        The width of the side borders to be masked.
    shape : tuple
        The shape (height, width) of the mask array to be generated.

    Returns
    -------
    np.ndarray
        The 2D mask array with specified borders masked (values set to 0)
        and the central area unmasked (values set to 1).
    """
    mask = np.ones(shape)
    mask[:distance_top, :] = 0
    mask[-distance_bottom:, :] = 0
    if distance_side > 0:
        mask[:, :distance_side] = 0
        mask[:, -distance_side:] = 0
    return mask
