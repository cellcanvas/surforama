from typing import TYPE_CHECKING, Optional

import napari
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QGroupBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from surforama import QtSurforama
from surforama.constants import (
    NAPARI_NORMAL_0,
    NAPARI_NORMAL_1,
    NAPARI_NORMAL_2,
    NAPARI_UP_0,
    NAPARI_UP_1,
    NAPARI_UP_2,
    ROTATION,
)
from surforama.utils.geometry import rotate_around_vector
from surforama.utils.napari import (
    update_rotations_on_points_layer,
    vectors_data_from_points_layer,
)


class QtSurfacePicker(QGroupBox):
    ENABLE_BUTTON_TEXT = "Enable"
    DISABLE_BUTTON_TEXT = "Disable"

    def __init__(
        self, surforama: "QtSurforama", parent: Optional[QWidget] = None
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
        new_rotations = rotation_radians * np.ones(
            len(selected_points), dtype=float
        )

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
        self.points_layer.events.data.connect(self._on_points_update)
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

    def _on_points_update(self, event=None):
        """Update the vectors layers when the points data are updated."""
        # update the vectors
        normal_data, up_data = vectors_data_from_points_layer(
            self.points_layer
        )
        self.normal_vectors_layer.data = normal_data
        self.up_vectors_layer.data = up_data

        # colors were being reset - this might not be necessary
        self.normal_vectors_layer.edge_color = "purple"
        self.up_vectors_layer.edge_color = "orange"

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
        table_defaults[ROTATION] = 0.0
        self.normal_vectors = np.concatenate(
            (self.normal_vectors, np.atleast_2d(normal_vector))
        )
        self.up_vectors = np.concatenate(
            (self.up_vectors, np.atleast_2d(up_vector))
        )
        self.rotations = np.append(self.rotations, 0)

        with self.points_layer.events.data.blocker(self._on_points_update):
            # we block since the event emission is before the features are updated.
            self.points_layer.add(np.atleast_2d(intersection_coords))
        self._on_points_update()
