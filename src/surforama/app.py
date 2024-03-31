from typing import List, Optional

import mrcfile
import napari
import numpy as np
import trimesh
from magicgui import magicgui
from napari.layers import Image, Surface
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import map_coordinates

from surforama.constants import (
    NAPARI_NORMAL_0,
    NAPARI_NORMAL_1,
    NAPARI_NORMAL_2,
    NAPARI_UP_0,
    NAPARI_UP_1,
    NAPARI_UP_2,
    ROTATION,
)
from surforama.gui.qt_mesh_generator import QtMeshGenerator
from surforama.gui.qt_point_io import QtPointIO
from surforama.io import read_obj_file
from surforama.utils.geometry import rotate_around_vector
from surforama.utils.napari import (
    update_rotations_on_points_layer,
    vectors_data_from_points_layer,
)


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
        self.slider_value = QLabel("0 vx", self)

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
        self.layout().addWidget(QLabel("Extend/contract surface"))
        self.layout().addLayout(self.sliderLayout)
        self.layout().addWidget(QLabel("Surface Thickness"))
        self.layout().addLayout(self.sampling_depth_sliderLayout)
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
        self.volume = image_layer.data.astype(np.float32)

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


class QtSurfacePicker(QGroupBox):
    ENABLE_BUTTON_TEXT = "Enable"
    DISABLE_BUTTON_TEXT = "Disable"

    def __init__(
        self, surforama: QtSurforama, parent: Optional[QWidget] = None
    ):
        super().__init__("Pick on surface", parent=parent)
        self.surforama = surforama
        self.points_layer = None
        self.normal_vectors_layer = None

        # initialize orientation data
        # todo store elsewhere (e.g., layer features)
        self.normal_vectors = np.empty((0, 3))
        self.up_vectors = np.empty((0, 3))
        self.rotations = np.empty((0,))

        # enable state
        self.enabled = False

        # make the activate button
        self.enable_button = QPushButton(self.ENABLE_BUTTON_TEXT)
        self.enable_button.clicked.connect(self._on_enable_button_pressed)

        # make the rotation slider
        self.rotation_slider = QSlider()
        self.rotation_slider.setOrientation(Qt.Horizontal)
        self.rotation_slider.setMinimum(-180)
        self.rotation_slider.setMaximum(180)
        self.rotation_slider.setValue(0)
        self.rotation_slider.valueChanged.connect(self._update_rotation)

        # make the layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.enable_button)
        self.layout().addWidget(self.rotation_slider)

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
            if self.normal_vectors_layer is None:
                self._initialize_vectors_layers()
            self.points_layer.visible = True

        self._on_enable_change()

    def _on_enable_change(self):
        if self.enabled:
            self._connect_mouse_callbacks()
        else:
            self._disconnect_mouse_callbacks()

    def _update_rotation(self, value):
        """Callback function to update the rotation of the selected points."""
        selected_points = list(self.points_layer.selected_data)
        self.rotations[selected_points] = value

        rotation_radians = value * (np.pi / 180)
        new_rotations = rotation_radians * np.ones(len(selected_points))

        old_up_vector = self.up_vectors[selected_points]
        normal_vector = self.normal_vectors[selected_points]

        new_up_vector = rotate_around_vector(
            rotate_around=normal_vector,
            to_rotate=old_up_vector,
            angle=rotation_radians,
        )

        update_rotations_on_points_layer(
            points_layer=self.points_layer,
            point_index=selected_points,
            rotations=new_rotations,
        )

        self.up_vectors_layer.data[selected_points, 1, :] = new_up_vector
        self.up_vectors_layer.refresh()

    def _initialize_points_layer(self):
        self.points_layer = self.surforama.viewer.add_points(
            ndim=3, size=3, face_color="magenta"
        )
        self.points_layer.shading = "spherical"
        self.surforama.viewer.layers.selection = [self.surforama.surface_layer]

    def _initialize_vectors_layers(self):
        self.normal_vectors_layer = self.surforama.viewer.add_vectors(
            ndim=3,
            length=10,
            edge_color="cornflowerblue",
            name="surface normals",
        )
        self.up_vectors_layer = self.surforama.viewer.add_vectors(
            ndim=3, length=10, edge_color="orange", name="up vectors"
        )
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

        # get the intersection point
        candidate_vertices = layer.data[1][triangle_index]
        candidate_points = layer.data[0][candidate_vertices]
        (
            _,
            intersection_coords,
        ) = napari.utils.geometry.find_nearest_triangle_intersection(
            event.position, event.view_direction, candidate_points[None, :, :]
        )

        # get normal vector of intersected triangle
        mesh = self.surforama.mesh
        normal_vector = mesh.face_normals[triangle_index]

        # create the orientation coordinate system
        up_vector = np.cross(
            normal_vector, [1, 0, 0]
        )  # todo add check if normal is parallel

        # store the data
        feature_table = self.points_layer._feature_table
        table_defaults = feature_table.defaults
        table_defaults[NAPARI_NORMAL_0] = normal_vector[0]
        table_defaults[NAPARI_NORMAL_1] = normal_vector[1]
        table_defaults[NAPARI_NORMAL_2] = normal_vector[2]
        table_defaults[NAPARI_UP_0] = up_vector[0]
        table_defaults[NAPARI_UP_1] = up_vector[1]
        table_defaults[NAPARI_UP_2] = up_vector[2]
        table_defaults[ROTATION] = 0
        self.normal_vectors = np.concatenate(
            (self.normal_vectors, np.atleast_2d(normal_vector))
        )
        self.up_vectors = np.concatenate(
            (self.up_vectors, np.atleast_2d(up_vector))
        )
        self.rotations = np.append(self.rotations, 0)

        self.points_layer.add(np.atleast_2d(intersection_coords))

        # update the vectors
        normal_data, up_data = vectors_data_from_points_layer(
            self.points_layer
        )
        self.normal_vectors_layer.data = normal_data
        self.up_vectors_layer.data = up_data

        # colors were being reset - this might not be necessary
        self.normal_vectors_layer.edge_color = "purple"
        self.up_vectors_layer.edge_color = "orange"


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
