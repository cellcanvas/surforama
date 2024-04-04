from typing import List, Optional

import mrcfile
import napari
import numpy as np
import trimesh
from magicgui import magicgui
from napari.layers import Image, Surface
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import map_coordinates

from surforama.gui.qt_mesh_generator import QtMeshGenerator
from surforama.gui.qt_point_io import QtPointIO
from surforama.gui.qt_surface_picker import QtSurfacePicker
from surforama.io import read_obj_file


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
        self._enabled = False

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
        self.slider_value = QLabel("0 vx", self)
        self.slider_value.setVisible(False)
        self.slider_title = QLabel("Extend/contract surface")
        self.slider_title.setVisible(False)

        self.sliderLayout = QHBoxLayout()
        self.sliderLayout.addWidget(self.slider)
        self.sliderLayout.addWidget(self.slider_value)

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
        self.sampling_depth_value = QLabel("10", self)
        self.sampling_depth_value.setVisible(False)
        self.sampling_depth_title = QLabel("Surface Thickness")
        self.sampling_depth_title.setVisible(False)

        self.sampling_depth_sliderLayout = QHBoxLayout()
        self.sampling_depth_sliderLayout.addWidget(self.sampling_depth_slider)
        self.sampling_depth_sliderLayout.addWidget(self.sampling_depth_value)

        # make the picking widget
        self.picking_widget = QtSurfacePicker(surforama=self, parent=self)
        self.picking_widget.setVisible(False)

        # make the saving widget
        self.point_writer_widget = QtPointIO(
            surface_picker=self.picking_widget, parent=self
        )
        self.point_writer_widget.setVisible(False)

        self.mesh_generator_widget = QtMeshGenerator(viewer, parent=self)

        # make the layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.mesh_generator_widget)
        self.layout().addWidget(self._layer_selection_widget.native)
        # self.layout().addWidget(QLabel("Extend/contract surface"))
        self.layout().addWidget(self.slider_title)
        self.layout().addLayout(self.sliderLayout)
        # self.layout().addWidget(QLabel("Surface Thickness"))
        self.layout().addWidget(self.sampling_depth_title)
        self.layout().addLayout(self.sampling_depth_sliderLayout)
        self.layout().addWidget(self.picking_widget)
        self.layout().addWidget(self.point_writer_widget)
        self.layout().addStretch()

        # set the layers
        self._set_layers(surface_layer=surface_layer, image_layer=volume_layer)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, enabled: bool):
        if enabled == self.enabled:
            # no change
            return

        if enabled:
            # make the widgets visible
            self.slider.setVisible(True)
            self.sampling_depth_slider.setVisible(True)
            self.picking_widget.setVisible(True)
            self.point_writer_widget.setVisible(True)
            self.slider_title.setVisible(True)
            self.slider_value.setVisible(True)
            self.sampling_depth_title.setVisible(True)
            self.sampling_depth_value.setVisible(True)
        else:
            # make the widgets visible
            self.slider.setVisible(False)
            self.sampling_depth_slider.setVisible(False)
            self.picking_widget.setVisible(False)
            self.point_writer_widget.setVisible(False)
            self.slider_title.setVisible(False)
            self.slider_value.setVisible(False)
            self.sampling_depth_title.setVisible(False)
            self.sampling_depth_value.setVisible(False)

        self._enabled = enabled

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
        self.surface_layer.shading = "none"
        self.surface_layer.refresh()

        self.normals = self.mesh.vertex_normals
        self.volume = image_layer.data.astype(np.float32)

        self.enabled = True

    def _get_valid_surface_layers(self, combo_box) -> List[Surface]:
        return [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, napari.layers.Surface)
        ]

    def _on_layer_update(self, event=None):
        """When the model updates the selected layer, update widgets."""
        self._layer_selection_widget.reset_choices()

        # check if the stored layers are still around
        layer_deleted = False
        if (
            self.surface_layer is not None
        ) and self.surface_layer not in self.viewer.layers:
            # remove the surface layer if it has been deleted.
            self.surface_layer = None
            layer_deleted = True

        if (self.image_layer is not None) and (
            self.image_layer not in self.viewer.layers
        ):
            # remove the surface layer if it has been deleted.
            self.image_layer = None
            layer_deleted = True

        if layer_deleted:
            self.enabled = False

    def _get_valid_image_layers(self, combo_box) -> List[Image]:
        return [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, napari.layers.Image)
        ]

    def get_point_colors(self, points):
        point_values = map_coordinates(
            self.volume, points.T, order=1, mode="nearest"
        )

        normalized_values = (point_values - point_values.min()) / (
            point_values.max() - point_values.min() + np.finfo(float).eps
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

        new_positions[:, 0] = np.clip(new_positions[:, 0], 0, vol_shape[0] - 1)
        new_positions[:, 1] = np.clip(new_positions[:, 1], 0, vol_shape[1] - 1)
        new_positions[:, 2] = np.clip(new_positions[:, 2], 0, vol_shape[2] - 1)

        self.color_values = new_colors
        self.vertices = new_positions
        self.update_mesh()
        self.slider_value.setText(f"{shift} vx")

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
        self.sampling_depth_value.setText(f"{value}")


if __name__ == "__main__":

    obj_path = "../../examples/tomo_17_M10_grow1_1_mesh_data.obj"
    tomo_path = "../../examples/tomo_17_M10_grow1_1_mesh_data.mrc"

    mrc = mrcfile.open(tomo_path, permissive=True)
    tomo_mrc = np.array(mrc.data)

    vertices, faces, values = read_obj_file(obj_path)
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
