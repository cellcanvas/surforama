from pathlib import Path
from typing import Optional

from magicgui import magicgui
from qtpy.QtWidgets import QGroupBox, QVBoxLayout, QWidget

from surforama.app import QtSurfacePicker
from surforama.io.star import oriented_points_to_star_file


class QtPointIO(QGroupBox):
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
        oriented_points_to_star_file(
            points_layer=self.surface_picker.points_layer,
            output_path=output_path,
        )
