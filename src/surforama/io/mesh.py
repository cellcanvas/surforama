import numpy as np
import trimesh


def read_obj_file(file_path):
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
