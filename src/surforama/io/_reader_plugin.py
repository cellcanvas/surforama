from typing import Callable, List

from napari.types import LayerDataTuple

from surforama.io import read_obj_file


def mesh_reader_plugin(path: str) -> Callable[[str], LayerDataTuple]:
    if isinstance(path, str) and path.endswith(".obj"):
        return obj_reader

    # format not recognized
    return None


def obj_reader(path: str) -> List[LayerDataTuple]:
    mesh_data = read_obj_file(path)
    return [(mesh_data, {}, "surface")]
