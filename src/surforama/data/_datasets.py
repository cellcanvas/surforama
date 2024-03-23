from typing import List, Tuple

import mrcfile
import numpy as np
import pooch
from napari.types import LayerDataTuple

from surforama.io.mesh import read_obj_file

_thylakoid_registry = pooch.create(
    path=pooch.os_cache("cellcanvas"),
    base_url="doi:10.5281/zenodo.10814409",
    registry={
        "S1_M3b_StII_grow2_1_mesh_data.mrc": "md5:a6e34bbf4edc767aa6c2c854c81c9c97",
        "S1_M3b_StII_grow2_1_mesh_data.obj": "md5:63b7d681204d7d3a3937154a0f4d7fc1",
        "S1_M3b_StII_grow2_1_mesh_data_seg.mrc": "md5:d88460eb3bdf3164be6053d281fc45be",
        "S1_M3c_StOI_grow2_1_mesh_data.mrc": "md5:296fbc48917c2baab7784a5ede6aae70",
        "S1_M3c_StOI_grow2_1_mesh_data.obj": "md5:076e6e8a825f67a24e28beba09edcf70",
        "S1_M3c_StOI_grow2_1_mesh_data_seg.mrc": "md5:878d4b3fc076dfc01e788cc08f9c9201",
    },
)

_covid_registry = pooch.create(
    path=pooch.os_cache("cellcanvas"),
    base_url="doi:10.5281/zenodo.10837518",
    registry={
        "TS_004_dose-filt_lp50_bin8.rec": "md5:31914b3aa32c5656acf583f08a581f64",
        "TS_004_dose-filt_lp50_bin8_membrain_model.obj": "md5:9e67ab9493096ce6455e244ca6b20220",
    },
)


def thylakoid_membrane() -> (
    Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]
):
    """Fetch the thylakoid membrane sample data.

    Data originally from Wietrzynski and Schaffer et al., eLife, 2020.
    https://doi.org/10.7554/eLife.53740
    """
    # get the tomogram
    tomogram_path = _thylakoid_registry.fetch(
        "S1_M3b_StII_grow2_1_mesh_data.mrc", progressbar=True
    )
    tomogram = mrcfile.read(tomogram_path).astype(float)

    # get the mesh
    mesh_path = _thylakoid_registry.fetch(
        "S1_M3b_StII_grow2_1_mesh_data.obj", progressbar=True
    )
    mesh_data = read_obj_file(mesh_path)

    return tomogram, mesh_data


def _thylakoid_membrane_sample_data_plugin() -> List[LayerDataTuple]:
    """napari sample data plugin for thylakoid membrane dataset."""

    # get the data
    tomogram, mesh_data = thylakoid_membrane()

    return [
        (tomogram, {"name": "tomogram"}, "image"),
        (mesh_data, {"name": "mesh"}, "surface"),
    ]


def covid_membrane() -> (
    Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]
):
    """Fetch the covid membrane sample data.

    Data originally from Ke et al., Nature, 2020.
    https://doi.org/10.5281/zenodo.10837519
    """
    # get the tomogram
    tomogram_path = _covid_registry.fetch(
        "TS_004_dose-filt_lp50_bin8.rec", progressbar=True
    )
    tomogram = mrcfile.read(tomogram_path).astype(float)

    # get the mesh
    mesh_path = _covid_registry.fetch(
        "TS_004_dose-filt_lp50_bin8_membrain_model.obj", progressbar=True
    )
    mesh_data = read_obj_file(mesh_path)

    return tomogram, mesh_data


def _covid_membrane_sample_data_plugin() -> List[LayerDataTuple]:
    """napari sample data plugin for thylakoid membrane dataset."""

    # get the data
    tomogram, mesh_data = covid_membrane()

    return [
        (tomogram, {"name": "tomogram"}, "image"),
        (mesh_data, {"name": "mesh"}, "surface"),
    ]
