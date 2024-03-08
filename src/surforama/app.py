from pathlib import Path
from typing import List, Optional

import mrcfile
import napari
import numpy as np
import pandas as pd
import starfile
import trimesh
from magicgui import magicgui
from napari.layers import Image, Surface
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QGroupBox,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


def read_obj_file_and_compute_normals(file_path, scale_factor=1):
    mesh = trimesh.load(file_path, file_type="obj", process=True)

    # Subdivide
    # verts, faces = trimesh.remesh.subdivide_to_size(
    # mesh.vertices, mesh.faces, 1
    # )

    # Subdivide can introduce holes
    # mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    # trimesh.repair.fill_holes(mesh)

    verts = mesh.vertices
    faces = mesh.faces

    verts = verts[:, [2, 1, 0]]

    values = np.ones((len(verts),))

    return verts, faces, values


# column names for the starfile
STAR_X_COLUMN_NAME = "rlnCoordinateX"
STAR_Y_COLUMN_NAME = "rlnCoordinateY"
STAR_Z_COLUMN_NAME = "rlnCoordinateZ"


class QtSurforama(QWidget):
    def __init__(
        self,
        viewer: napari.Viewer,
        surface_layer: Optional[Surface] = None,
        volume_layer: Optional[Image] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent=parent)
        self.viewer = viewer

        # make the layer selection widget
        self._layer_selection_widget = magicgui(
            self._set_layers,
            surface_layer={"choices": self._get_valid_surface_layers},
            image_layer={"choices": self._get_valid_image_layers},
            call_button="start surfing",
        )

        # add callback to update choices
        self.viewer.layers.events.inserted.connect(self._on_layer_update)
        self.viewer.layers.events.removed.connect(self._on_layer_update)

        # make the slider to change thickness
        self.slider = QSlider()
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setMinimum(-100)
        self.slider.setMaximum(100)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.slide_points)
        self.slider.setVisible(False)

        # New slider for sampling depth

        self.sampling_depth_slider = QSlider()
        self.sampling_depth_slider.setOrientation(Qt.Horizontal)
        self.sampling_depth_slider.setMinimum(1)
        self.sampling_depth_slider.setMaximum(100)
        self.sampling_depth_slider.setValue(10)
        self.sampling_depth_slider.valueChanged.connect(
            self.update_colors_based_on_sampling
        )
        self.sampling_depth_slider.setVisible(False)

        # make the picking widget
        self.picking_widget = QtSurfacePicker(surforama=self, parent=self)
        self.picking_widget.setVisible(False)

        # make the saving widget
        self.point_writer_widget = QtPointWriter(
            surface_picker=self.picking_widget, parent=self
        )
        self.point_writer_widget.setVisible(False)

        # make the layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._layer_selection_widget.native)
        self.layout().addWidget(QLabel("Extend/contract surface"))
        self.layout().addWidget(self.slider)
        self.layout().addWidget(QLabel("Surface Thickness"))
        self.layout().addWidget(self.sampling_depth_slider)
        self.layout().addWidget(self.picking_widget)
        self.layout().addWidget(self.point_writer_widget)
        self.layout().addStretch()

        # set the layers
        self._set_layers(surface_layer=surface_layer, image_layer=volume_layer)

    def _set_layers(
        self,
        surface_layer: Surface,
        image_layer: Image,
    ):
        self.surface_layer = surface_layer
        self.image_layer = image_layer

        if (surface_layer is None) or (image_layer is None):
            return

        # Create a mesh object using trimesh
        self.mesh = trimesh.Trimesh(
            vertices=surface_layer.data[0], faces=surface_layer.data[1]
        )

        self.vertices = self.mesh.vertices
        self.faces = self.mesh.faces

        # Compute vertex normals
        self.color_values = np.ones((self.mesh.vertices.shape[0],))

        self.surface_layer.data = (
            self.vertices,
            self.faces,
            self.color_values,
        )
        self.surface_layer.refresh()

        self.normals = self.mesh.vertex_normals
        self.volume = image_layer.data

        # make the widgets visible
        self.slider.setVisible(True)
        self.sampling_depth_slider.setVisible(True)
        self.picking_widget.setVisible(True)
        self.point_writer_widget.setVisible(True)

    def _get_valid_surface_layers(self, combo_box) -> List[Surface]:
        return [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, napari.layers.Surface)
        ]

    def _on_layer_update(self, event=None):
        """When the model updates the selected layer, update widgets."""
        self._layer_selection_widget.reset_choices()

    def _get_valid_image_layers(self, combo_box) -> List[Image]:
        return [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, napari.layers.Image)
        ]

    def get_point_colors(self, points):
        point_indices = points.astype(int)

        point_values = self.volume[
            point_indices[:, 0], point_indices[:, 1], point_indices[:, 2]
        ]
        normalized_values = (point_values - point_values.min()) / (
            point_values.max() - point_values.min()
        )

        return normalized_values

    def get_point_set(self):
        return self.mesh.vertices

    def get_faces(self):
        return self.mesh.faces

    def slide_points(self, value):
        # Calculate the new positions of points along their normals
        shift = value / 10
        new_positions = self.get_point_set() + (self.normals * shift)
        # Update the points layer with new positions
        new_colors = self.get_point_colors(new_positions)

        vol_shape = self.volume.shape

        new_positions[:, 0] = np.clip(new_positions[:, 0], 0, vol_shape[2] - 1)
        new_positions[:, 1] = np.clip(new_positions[:, 1], 0, vol_shape[1] - 1)
        new_positions[:, 2] = np.clip(new_positions[:, 2], 0, vol_shape[0] - 1)

        self.color_values = new_colors
        self.vertices = new_positions
        self.update_mesh()

    def update_mesh(self):
        self.surface_layer.data = (
            self.vertices,
            self.get_faces(),
            self.color_values,
        )

    def update_colors_based_on_sampling(self, value):
        spacing = 0.5
        sampling_depth = value / 10

        # Collect all samples for normalization calculation
        all_samples = []

        # Sample along the normal for each point
        for point, normal in zip(self.get_point_set(), self.normals):
            for depth in range(int(sampling_depth)):
                sample_point = point + normal * spacing * depth
                sample_point_clipped = np.clip(
                    sample_point, [0, 0, 0], np.array(self.volume.shape) - 1
                ).astype(int)
                sample_value = self.volume[
                    sample_point_clipped[0],
                    sample_point_clipped[1],
                    sample_point_clipped[2],
                ]
                all_samples.append(sample_value)

        # Calculate min and max across all sampled values
        samples_min = np.min(all_samples)
        samples_max = np.max(all_samples)

        # Normalize and update colors based on the mean value
        # of samples for each point
        new_colors = np.zeros((len(self.get_point_set()),))
        for i, (point, normal) in enumerate(
            zip(self.get_point_set(), self.normals)
        ):
            samples = []
            for depth in range(int(sampling_depth)):
                sample_point = point + normal * spacing * depth
                sample_point_clipped = np.clip(
                    sample_point, [0, 0, 0], np.array(self.volume.shape) - 1
                ).astype(int)
                sample_value = self.volume[
                    sample_point_clipped[0],
                    sample_point_clipped[1],
                    sample_point_clipped[2],
                ]
                samples.append(sample_value)

            # Normalize the mean of samples for this point using
            # the min and max from all samples
            mean_value = np.mean(samples)
            normalized_value = (
                (mean_value - samples_min) / (samples_max - samples_min)
                if samples_max > samples_min
                else 0
            )
            new_colors[i] = normalized_value

        self.color_values = new_colors
        self.update_mesh()


class QtSurfacePicker(QGroupBox):
    ENABLE_BUTTON_TEXT = "Enable"
    DISABLE_BUTTON_TEXT = "Disable"

    def __init__(
        self, surforama: QtSurforama, parent: Optional[QWidget] = None
    ):
        super().__init__("Pick on surface", parent=parent)
        self.surforama = surforama
        self.points_layer = None

        # enable state
        self.enabled = False

        # make the activate button
        self.enable_button = QPushButton(self.ENABLE_BUTTON_TEXT)
        self.enable_button.clicked.connect(self._on_enable_button_pressed)

        # make the layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.enable_button)

    def _on_enable_button_pressed(self, event):
        # toggle enabled
        if self.enabled:
            # if currently enabled, toggle to disabled
            self.enabled = False
            self.enable_button.setText(self.ENABLE_BUTTON_TEXT)

        else:
            # if disabled, toggle to enabled
            self.enabled = True
            self.enable_button.setText(self.DISABLE_BUTTON_TEXT)

            if self.points_layer is None:
                self._initialize_points_layer()
            self.points_layer.visible = True

        self._on_enable_change()

    def _on_enable_change(self):
        if self.enabled:
            self._connect_mouse_callbacks()
        else:
            self._disconnect_mouse_callbacks()

    def _initialize_points_layer(self):
        self.points_layer = self.surforama.viewer.add_points(
            ndim=3, size=3, face_color="magenta"
        )
        self.points_layer.shading = "spherical"
        self.surforama.viewer.layers.selection = [self.surforama.surface_layer]

    def _connect_mouse_callbacks(self):
        self.surforama.surface_layer.mouse_drag_callbacks.append(
            self._find_point_on_click
        )

    def _disconnect_mouse_callbacks(self):
        self.surforama.surface_layer.mouse_drag_callbacks.remove(
            self._find_point_on_click
        )

    def _find_point_on_click(self, layer, event):
        # if "Alt" not in event.modifiers:
        #    return
        value = layer.get_value(
            event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True,
        )
        if value is None:
            return
        triangle_index = value[1]
        if triangle_index is None:
            # if the click did not intersect the mesh, don't do anything
            return

        candidate_vertices = layer.data[1][triangle_index]
        candidate_points = layer.data[0][candidate_vertices]
        (
            _,
            intersection_coords,
        ) = napari.utils.geometry.find_nearest_triangle_intersection(
            event.position, event.view_direction, candidate_points[None, :, :]
        )

        self.points_layer.add(np.atleast_2d(intersection_coords))


class QtPointWriter(QGroupBox):
    def __init__(
        self, surface_picker: QtSurfacePicker, parent: Optional[QWidget] = None
    ):
        super().__init__("Save points", parent=parent)
        self.surface_picker = surface_picker

        # make the points saving widget
        self.file_saving_widget = magicgui(
            self._write_star_file,
            output_path={"mode": "w"},
            call_button="Save to star file",
        )

        # make the layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.file_saving_widget.native)

    def _write_star_file(self, output_path: Path):
        points = self.surface_picker.points_layer.data
        points_table = pd.DataFrame(
            {
                STAR_Z_COLUMN_NAME: points[:, 0],
                STAR_Y_COLUMN_NAME: points[:, 1],
                STAR_X_COLUMN_NAME: points[:, 2],
            }
        )
        starfile.write(points_table, output_path)


if __name__ == "__main__":

    obj_path = "tomo_17_M10_grow1_1_mesh_data.obj"
    tomo_path = "tomo_17_M10_grow1_1_mesh_data.mrc"

    mrc = mrcfile.open(tomo_path)
    tomo_mrc = np.array(mrc.data)

    vertices, faces, values = read_obj_file_and_compute_normals(obj_path)
    surface = (vertices, faces, values)

    viewer = napari.Viewer(ndisplay=3)
    volume_layer = viewer.add_image(tomo_mrc)
    surface_layer = viewer.add_surface(surface)

    # Testing points

    point_set = surface[0]

    volume_shape = np.array(tomo_mrc.data.shape)
    points_indices = np.round(point_set).astype(int)

    # Instantiate the widget and add it to Napari
    surforama_widget = QtSurforama(viewer, surface_layer, volume_layer)
    viewer.window.add_dock_widget(
        surforama_widget, area="right", name="Surforama"
    )

    napari.run()
