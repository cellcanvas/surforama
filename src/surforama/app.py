import numpy as np
import napari
import mrcfile
import trimesh

from qtpy.QtWidgets import QWidget, QVBoxLayout, QSlider, QLabel
from qtpy.QtCore import Qt

def read_obj_file_and_compute_normals(file_path, scale_factor=1):
    mesh = trimesh.load(file_path, file_type='obj', process=True)
    
    # Subdivide
    # verts, faces = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, 1)    

    # Subdivide can introduce holes
    # mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    # trimesh.repair.fill_holes(mesh)
    
    verts = mesh.vertices
    faces = mesh.faces

    verts = verts[:, [2, 1, 0]]
    
    values = np.ones((len(verts),))

    return verts, faces, values

class Surforama(QWidget):
    def __init__(self, viewer, surface_layer, volume_layer):
        super().__init__()
        self.viewer = viewer

        self.surface_layer = surface_layer
        self.volume_layer = volume_layer

        # Create a mesh object using trimesh
        self.mesh = trimesh.Trimesh(vertices=surface_layer.data[0], faces=surface_layer.data[1])

        self.vertices = self.mesh.vertices
        self.faces = self.mesh.faces
        
        # Compute vertex normals
        self.color_values = np.ones((self.mesh.vertices.shape[0],))

        self.surface_layer.data = (self.vertices, self.faces, self.color_values)
                
        self.normals = self.mesh.vertex_normals
        self.volume = volume_layer.data
        
        self.layout = QVBoxLayout()
        label = QLabel("Extend/contract surface")
        self.layout.addWidget(label)
        self.slider = QSlider()
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setMinimum(-100)
        self.slider.setMaximum(100)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.slide_points)
        self.layout.addWidget(self.slider)

        # New slider for sampling depth
        label = QLabel("Surface Thickness")
        self.layout.addWidget(label)
        self.sampling_depth_slider = QSlider()
        self.sampling_depth_slider.setOrientation(Qt.Horizontal)
        self.sampling_depth_slider.setMinimum(1)
        self.sampling_depth_slider.setMaximum(100)
        self.sampling_depth_slider.setValue(10)
        self.sampling_depth_slider.valueChanged.connect(self.update_colors_based_on_sampling)
        self.layout.addWidget(self.sampling_depth_slider)
        
        self.setLayout(self.layout)

    def get_point_colors(self, points):
        point_indices = points.astype(int)

        point_values = self.volume[point_indices[:, 0], point_indices[:, 1], point_indices[:, 2]]
        normalized_values = (point_values - point_values.min()) / (point_values.max() - point_values.min())

        return normalized_values

    def get_point_set(self):
        return self.vertices
    
    def slide_points(self, value):
        # Calculate the new positions of points along their normals
        shift = value / 10
        new_positions = self.get_point_set() + (self.normals * shift)
        # Update the points layer with new positions
        new_colors = self.get_point_colors(new_positions)

        self.color_values = new_colors
        self.vertices = new_positions
        self.update_mesh()

    def get_faces(self):
        return self.faces

    def update_mesh(self):
        self.surface_layer.data = (self.get_point_set(), self.get_faces(), self.color_values)

    def update_colors_based_on_sampling(self, value):
        spacing = 0.5
        sampling_depth = value / 10

        # Collect all samples for normalization calculation
        all_samples = []

        # Sample along the normal for each point
        for point, normal in zip(self.get_point_set(), self.normals):
            for depth in range(int(sampling_depth)):
                sample_point = point + normal * spacing * depth
                sample_point_clipped = np.clip(sample_point, [0, 0, 0], np.array(self.volume.shape) - 1).astype(int)
                sample_value = self.volume[sample_point_clipped[0], sample_point_clipped[1], sample_point_clipped[2]]
                all_samples.append(sample_value)

        # Calculate min and max across all sampled values
        samples_min = np.min(all_samples)
        samples_max = np.max(all_samples)

        # Normalize and update colors based on the mean value of samples for each point
        new_colors = np.zeros((len(self.get_point_set()),))
        for i, (point, normal) in enumerate(zip(self.get_point_set(), self.normals)):
            samples = []
            for depth in range(int(sampling_depth)):
                sample_point = point + normal * spacing * depth
                sample_point_clipped = np.clip(sample_point, [0, 0, 0], np.array(self.volume.shape) - 1).astype(int)
                sample_value = self.volume[sample_point_clipped[0], sample_point_clipped[1], sample_point_clipped[2]]
                samples.append(sample_value)

            # Normalize the mean of samples for this point using the min and max from all samples
            mean_value = np.mean(samples)
            normalized_value = (mean_value - samples_min) / (samples_max - samples_min) if samples_max > samples_min else 0
            new_colors[i] = normalized_value

        self.color_values = new_colors
        self.update_mesh()

        
if __name__ == "__main__":
    # obj_path = "/Users/kharrington/Data/membranorama/T17S1C3M4.obj"
    # tomo_path = "/Users/kharrington/Data/membranorama/tomo17_load1G5L3_bin4_denoised_ctfcorr_scaled3.rec"

    # obj_path = "/Users/kharrington/Data/membranorama/TS_004_dose-filt_lp50_bin8_membrain_model.obj"
    # tomo_path = "/Users/kharrington/Data/membranorama/TS_004_dose-filt_lp50_bin8.rec"

    obj_path = "/Users/kharrington/Data/membranorama/tomo_17_M10_grow1_1_mesh_data.obj"
    tomo_path = "/Users/kharrington/Data/membranorama/tomo_17_M10_grow1_1_mesh_data.mrc"

    mrc = mrcfile.open(tomo_path)
    tomo_mrc = np.array(mrc.data)

    # vertices, faces, values, normals = read_obj_file_and_compute_normals(obj_path, scale_factor=14.08)
    vertices, faces, values = read_obj_file_and_compute_normals(obj_path)
    surface = (vertices, faces, values)
    # print("Vertices:", vertices)
    # print("Faces:", faces)

    viewer = napari.Viewer(ndisplay=3)

    volume_layer = viewer.add_image(tomo_mrc)
    surface_layer = viewer.add_surface(surface)

    # Testing points

    point_set = surface[0]

    volume_shape = np.array(tomo_mrc.data.shape)
    points_indices = np.round(point_set).astype(int)

    # Instantiate the widget and add it to Napari
    surforama_widget = Surforama(viewer, surface_layer, volume_layer)
    viewer.window.add_dock_widget(surforama_widget, area='right', name='Surforama')
