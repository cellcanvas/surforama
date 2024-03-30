from typing import Optional

import typer
from typing_extensions import Annotated

app = typer.Typer(
    help="Surforama: wiew tomogram densities on a surface.",
    no_args_is_help=True,
)


@app.command()
def launch_surforama(
    image_path: Annotated[
        Optional[str], typer.Option(help="Path to the image to load.")
    ] = None,
    mesh_path: Annotated[
        Optional[str], typer.Option(help="Path to the mesh to load.")
    ] = None,
    demo: Annotated[
        bool, typer.Option("--demo", help="launch surforama with sample data")
    ] = False,
):
    if demo and (image_path is not None or mesh_path is not None):
        raise ValueError(
            "Please do not specify an image/mesh path when launching in demo mode."
        )

    # delay imports
    import mrcfile
    import napari

    from surforama.app import QtSurforama
    from surforama.io.mesh import read_obj_file

    viewer = napari.Viewer(ndisplay=3)

    if demo:
        # set up the viewer in demo mode
        from surforama.data._datasets import thylakoid_membrane

        # fetch the data
        tomogram, mesh_data = thylakoid_membrane()

        # Add the data to the viewer
        volume_layer = viewer.add_image(
            tomogram, blending="translucent", depiction="plane"
        )
        surface_layer = viewer.add_surface(mesh_data)

        # set up the slicing plane position
        volume_layer.plane = {"normal": [1, 0, 0], "position": [66, 187, 195]}

        # set up the camera
        viewer.camera.center = (64.0, 124.0, 127.5)
        viewer.camera.zoom = 3.87
        viewer.camera.angles = (
            -5.401480002668876,
            -0.16832643131442776,
            160.28901483338126,
        )

    else:
        # set up the viewer with the user-requested images
        if image_path is not None:
            # load the image if the path was passed
            image = mrcfile.read(image_path).astype(float)
            volume_layer = viewer.add_image(image)
        else:
            volume_layer = None

        if mesh_path is not None:
            # load the mesh if a path was passed
            mesh_data = read_obj_file(mesh_path)
            surface_layer = viewer.add_surface(mesh_data)
        else:
            surface_layer = None

    # Instantiate the widget and add it to Napari
    surforama_widget = QtSurforama(viewer, surface_layer, volume_layer)
    viewer.window.add_dock_widget(
        surforama_widget, area="right", name="Surforama"
    )

    napari.run()
