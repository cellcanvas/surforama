from typing import Optional

import napari
import mrcfile
import numpy as np
import typer
from typing_extensions import Annotated

from surforama.app import QtSurforama, read_obj_file_and_compute_normals

app = typer.Typer(
    help="Surforama: wiew tomogram densities on a surface.",
    no_args_is_help=True,
)


@app.command()
def launch_surforama(
    image_path: Annotated[
        Optional[str],
        typer.Argument(help="Path to the image to load.")
    ]=None,
    mesh_path: Annotated[
        Optional[str],
        typer.Argument(help="Path to the mesh to load.")
    ]=None
):
    viewer = napari.Viewer(ndisplay=3)
    if image_path is not None:
        # load the image if the path was passed
        image = mrcfile.read(image_path)
        volume_layer = viewer.add_image(image)
    else:
        volume_layer = None

    if mesh_path is not None:
        # load the mesh if a path was passed
        mesh_data = read_obj_file_and_compute_normals(mesh_path)
        surface_layer = viewer.add_surface(mesh_data)
    else:
        surface_layer = None

    # Instantiate the widget and add it to Napari
    surforama_widget = QtSurforama(viewer, surface_layer, volume_layer)
    viewer.window.add_dock_widget(surforama_widget, area='right', name='Surforama')

    napari.run()
