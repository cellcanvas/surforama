from pathlib import Path
from typing import TYPE_CHECKING, Optional

from magicgui import magicgui
from napari.utils.notifications import show_warning
from qtpy.QtWidgets import QGroupBox, QTabWidget, QVBoxLayout, QWidget

if TYPE_CHECKING:
    from surforama.app import QtSurfacePicker
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
        normal_data, up_data = vectors_data_from_points_data(
            point_coordinates=point_coordinates, features_table=features_table
        )

        # add the data to the viewer
        self.surface_picker.points_layer.data = point_coordinates
        self.surface_picker.points_layer.features = features_table

        self.surface_picker.normal_vectors_layer.data = normal_data
        self.surface_picker.up_vectors_layer.data = up_data

        self.surface_picker.normal_vectors_layer.edge_color = "purple"
        self.surface_picker.up_vectors_layer.edge_color = "orange"
