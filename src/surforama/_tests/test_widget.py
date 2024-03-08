import numpy as np

from surforama.app import QtSurforama


# capsys is a pytest fixture that captures stdout and stderr output streams
def test_surforama(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    image_layer = viewer.add_image(np.random.random((20, 20, 20)))
    surface_layer = viewer.add_surface(
        (
            np.random.random((100, 3)) * 20,
            (np.random.random((100, 3)) * 100).astype(int),
        )
    )

    # create our widget, passing in the viewer
    _ = QtSurforama(viewer, surface_layer, image_layer)
