from typing import List, Optional

import napari
from magicgui import magicgui
from napari.layers import Labels
from qtpy.QtWidgets import QGroupBox, QVBoxLayout, QWidget

from surforama.io import convert_mask_to_mesh


class QtMeshGenerator(QGroupBox):
    def __init__(
        self, viewer: napari.Viewer, parent: Optional[QWidget] = None
    ):
        super().__init__("Generate Mesh from Labels", parent=parent)
        self.viewer = viewer

        # make the labels layer selection widget
        self.labels_layer_selection_widget = magicgui(
            self._generate_mesh_from_labels,
            labels_layer={"choices": self._get_valid_labels_layers},
            barycentric_area={
                "widget_type": "Slider",
                "min": 0.1,
                "max": 10.0,
                "value": 1.0,
                "step": 0.1,
            },
            smoothing={
                "widget_type": "Slider",
                "min": 0,
                "max": 1000,
                "value": 1000,
            },
            call_button="Generate Mesh",
        )

        # make the layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.labels_layer_selection_widget.native)

        # Add callback to update choices when layers change
        self.viewer.layers.events.inserted.connect(self._on_layer_update)
        self.viewer.layers.events.removed.connect(self._on_layer_update)

    def _on_layer_update(self, event=None):
        """Refresh the layer choices when layers are added or removed."""
        self.labels_layer_selection_widget.reset_choices()

    def _get_valid_labels_layers(self, combo_box) -> List[Labels]:
        return [
            layer
            for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Labels)
        ]

    def _generate_mesh_from_labels(
        self,
        labels_layer: Labels,
        smoothing: int = 10,
        barycentric_area: float = 1.0,
    ):
        # Assuming create_mesh_from_mask exists and generates vertices, faces, and values
        vertices, faces, values = convert_mask_to_mesh(
            labels_layer.data,
            smoothing=smoothing,
            barycentric_area=barycentric_area,
        )
        self.viewer.add_surface((vertices, faces, values))
