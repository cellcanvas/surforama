from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from magicgui import magicgui
from napari.utils.notifications import show_warning
from qtpy.QtWidgets import QGroupBox, QTabWidget, QVBoxLayout, QWidget
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from surforama.gui.qt_surface_picker import QtSurfacePicker
from surforama.constants import (
    NAPARI_NORMAL_0,
    NAPARI_NORMAL_1,
    NAPARI_NORMAL_2,
    NAPARI_UP_0,
    NAPARI_UP_1,
    NAPARI_UP_2,
    ROTATION,
)
from surforama.io.star import (
    load_points_layer_data_from_star_file,
    oriented_points_to_star_file,
)
from surforama.utils.napari import vectors_data_from_points_data


class QtPointIO(QGroupBox):
    def __init__(
        self,
        surface_picker: "QtSurfacePicker",
        parent: Optional[QWidget] = None,
    ):
        super().__init__("Save points", parent=parent)
        self.surface_picker = surface_picker

        # make the points saving widget
        self.file_saving_widget = magicgui(
            self._write_star_file,
            file_path={"mode": "w"},
            call_button="Save to star file",
        )
        self.file_loading_widget = magicgui(
            self._load_star_file,
            file_path={"mode": "r"},
            call_button="Load from star file",
        )

        # make the tab widget
        self.tab_widget = QTabWidget(self)
        self.tab_widget.addTab(self.file_saving_widget.native, "save")
        self.tab_widget.addTab(self.file_loading_widget.native, "load")

        # make the layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.tab_widget)

    def _assign_orientations_from_nearest_triangles(
        self,
        point_coordinates: np.ndarray,
    ):
        """Assign orientations to the points based on the nearest mesh triangles.

        The closest triangle to each point is found and the normal and up vectors
        (perpendicular to the normal) are computed for each point based on the
        triangle normals.

        Parameters
        ----------
        point_coordinates : np.ndarray
            (n, 3) array of the coordinates of each point.
        """

        vertices = self.surface_picker.surforama.surface_layer.data[0]
        faces = self.surface_picker.surforama.surface_layer.data[1]
        triangle_centers = vertices[faces].mean(axis=1)

        # find closest triangle for each point
        distance_tree = cKDTree(triangle_centers)
        _, closest_triangle_indices = distance_tree.query(point_coordinates)

        # get the triangle normals via right-hand rule
        closest_triangles = faces[closest_triangle_indices]
        triangle_normals = np.cross(
            vertices[closest_triangles[:, 1]]
            - vertices[closest_triangles[:, 0]],
            vertices[closest_triangles[:, 2]]
            - vertices[closest_triangles[:, 0]],
        )
        triangle_normals /= np.linalg.norm(triangle_normals, axis=1)[:, None]

        # find any perpendicular vector per computed normal
        up_vectors = np.cross(triangle_normals, np.array([0, 0, 1]))
        up_vectors /= np.linalg.norm(up_vectors, axis=1)[:, None]

        n_points = point_coordinates.shape[0]
        normal_vector_data = np.zeros((n_points, 2, 3))
        normal_vector_data[:, 0, :] = point_coordinates
        normal_vector_data[:, 1, :] = triangle_normals

        up_vector_data = np.zeros((n_points, 2, 3))
        up_vector_data[:, 0, :] = point_coordinates
        up_vector_data[:, 1, :] = up_vectors

        return normal_vector_data, up_vector_data

    def _write_star_file(self, file_path: Path):
        oriented_points_to_star_file(
            points_layer=self.surface_picker.points_layer,
            output_path=file_path,
        )

    def _load_star_file(self, file_path: Path):
        """Load oriented points from a star file and add them
        to the viewer.
        """
        if self.surface_picker.enabled is False:
            show_warning("The surface picker must be enabled to load a file.")
            return
        # get the points data
        point_coordinates, features_table = (
            load_points_layer_data_from_star_file(file_path=file_path)
        )

        # get the vectors data
        if features_table is None:
            normal_data, up_data = (
                self._assign_orientations_from_nearest_triangles(
                    point_coordinates=point_coordinates
                )
            )
            features_table = {
                NAPARI_NORMAL_0: normal_data[:, 1, 0],
                NAPARI_NORMAL_1: normal_data[:, 1, 1],
                NAPARI_NORMAL_2: normal_data[:, 1, 2],
                NAPARI_UP_0: up_data[:, 1, 0],
                NAPARI_UP_1: up_data[:, 1, 1],
                NAPARI_UP_2: up_data[:, 1, 2],
                ROTATION: np.zeros(normal_data.shape[0]) * 1.0,
            }
        else:
            normal_data, up_data = vectors_data_from_points_data(
                point_coordinates=point_coordinates,
                features_table=features_table,
            )

        # add the data to the viewer
        self.surface_picker.points_layer.data = point_coordinates
        self.surface_picker.points_layer.features = features_table

        self.surface_picker.normal_vectors_layer.data = normal_data
        self.surface_picker.up_vectors_layer.data = up_data

        self.surface_picker.normal_vectors_layer.edge_color = "purple"
        self.surface_picker.up_vectors_layer.edge_color = "orange"

        self.surface_picker.rotations = features_table[ROTATION]
        self.surface_picker.up_vectors = up_data[:, 1, :]
        self.surface_picker.normal_vectors = normal_data[:, 1, :]
