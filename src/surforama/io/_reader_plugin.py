from typing import Callable, List

from napari.types import LayerDataTuple

from surforama.io import read_obj_file
from surforama.io.star import load_points_layer_data_from_star_file
from surforama.utils.napari import vectors_data_from_points_data


def mesh_reader_plugin(path: str) -> Callable[[str], LayerDataTuple]:
    if isinstance(path, str) and path.endswith(".obj"):
        return obj_reader

    # format not recognized
    return None


def obj_reader(path: str) -> List[LayerDataTuple]:
    mesh_data = read_obj_file(path)
    return [(mesh_data, {}, "surface")]


def star_reader_plugin(path: str) -> Callable[[str], LayerDataTuple]:
    if isinstance(path, str) and path.endswith(".star"):
        return star_reader

    # format not recognized
    return None


def star_reader(path: str) -> List[LayerDataTuple]:
    point_coordinates, features_table = load_points_layer_data_from_star_file(
        file_path=path
    )

    # make the points layer data
    points_layer_data = (
        point_coordinates,
        {"size": 3, "face_color": "magenta", "features": features_table},
        "points",
    )

    # make the vectors layer data
    normal_data, up_data = vectors_data_from_points_data(
        point_coordinates=point_coordinates, features_table=features_table
    )
    normal_layer_data = (
        normal_data,
        {"length": 10, "edge_color": "orange", "name": "normal vectors"},
        "vectors",
    )
    up_layer_data = (
        up_data,
        {"length": 10, "edge_color": "purple", "name": "up vectors"},
        "vectors",
    )

    return [points_layer_data, normal_layer_data, up_layer_data]
