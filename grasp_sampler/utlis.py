import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import TYPE_CHECKING, Union
import numpy.typing as npt

import yaml
from yaml import Loader
import os
import re
import argparse

from pxr import Usd, UsdGeom
import aspose.threed as a3d


def usd2obj(input_usd, output_obj) -> None:
    # load the USD in an object of Scene 
    scene = a3d.Scene.from_file(input_usd)
    # save USD as a OBJ 
    scene.save(output_obj, a3d.FileFormat.WAVEFRONT_OBJ)
    
def usd2trimesh(usd_file):
    """
    """
    stage = Usd.Stage.Open(usd_file)
    vertices = []
    faces = []

    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            vertices.extend(points)
            
            face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
            face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()

            # Convert face indices to face groups
            index_offset = 0
            for count in face_vertex_counts:
                # Assuming triangular faces
                if count == 3:
                    faces.append(face_vertex_indices[index_offset:index_offset + 3])
                index_offset += count

    # Create a trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh


def create_plane(scene_bounds):
    """
    Erzeugt eine horizontale Plane als sehr flache Box.
    Die Oberseite der Plane liegt auf scene_bounds[1][2].
    """

    bbox_min = scene_bounds[0]
    bbox_max = scene_bounds[1]

    width_x = bbox_max[0] - bbox_min[0]
    width_y = bbox_max[1] - bbox_min[1]

    height = 0.001
    scale = 1.5

    plane = trimesh.primitives.Box(extents=[width_x, width_y, height])
    plane.apply_scale(scale)

    transform = np.eye(4)

    # Mittelpunkt in x/y
    transform[0, 3] = (bbox_min[0] + bbox_max[0]) / 2.0
    transform[1, 3] = (bbox_min[1] + bbox_max[1]) / 2.0

    # Oberseite der Plane auf scene_bounds[1][2]
    scaled_height = height * scale
    transform[2, 3] = scene_bounds[0][2] - scaled_height / 2.0

    plane.apply_transform(transform)

    plane.visual.face_colors = np.array([0, 204, 255, 255])

    return plane


def print_options(args: argparse.Namespace) -> None:
    """Prints the current configuration in a formatted box."""
    options = {
        "Number of points to sample on the mesh": args.num_points,
        "Number of maximal grasps to sample": args.max_grasps,
        "Number of maximal rotations per grasp": args.max_rotations,
        "Friction coefficient": args.friction,
        "Visualize a single result": args.visualize_single,
        "Visualize all results": args.visualize_all,
    }

    title = "Configuration"
    key_width = max(len(k) for k in options)
    box_width = key_width + max(len(str(v)) for v in options.values()) + 7

    print("┌" + "─" * box_width + "┐")
    print(f"│  {title:^{box_width - 4}}  │")
    print("├" + "─" * box_width + "┤")
    for key, value in options.items():
        print(f"│  {key:<{key_width}} : {value!s:<{box_width - key_width - 5}} │")
    print("└" + "─" * box_width + "┘")


def as_matrix(seq: str, angles: Union[float, npt.ArrayLike], degrees: bool = ...):
    t = np.eye(4)
    t[:3, :3] = R.from_euler(seq, angles, degrees=degrees).as_matrix()
    
    return t


def load_yaml(file_path):
    if isinstance(file_path, str):
        with open(file_path) as file_p:
            yaml_params = yaml.load(file_p, Loader=Loader)
    else:
        yaml_params = file_path
    return yaml_params


def get_scene_paths(base_directory: str, indices: str = "") -> list:
    """
    Returns paths to 'scene.yaml' files in the base_directory.
    Indices can be e.g. '0-5' (range) or '1,3,5' (specific indices).
    """
    scene_paths = [
        os.path.join(root, 'scene.yaml')
        for root, dirs, files in os.walk(base_directory)
        if 'scene.yaml' in files
    ]
    print(scene_paths)
    scene_paths = sorted(
        scene_paths,
        key=lambda x: int(os.path.basename(os.path.dirname(x)).split('_')[-1])
    )

    if indices:
        # Range (e.g. 1-5)
        match = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", indices)
        if match:
            start, end = map(int, match.groups())
            return scene_paths[start:end]
        # Specific indices (e.g. 1,3,5)
        elif re.match(r"^\s*\d+(?:\s*,\s*\d+)*\s*$", indices):
            idx_list = [int(i.strip()) for i in indices.split(",")]
            return [scene_paths[i] for i in idx_list if 0 <= i < len(scene_paths)]
        else:
            raise ValueError("Invalid format. Allowed: 'start-end' or 'i1,i2,...'.")
    else:
        return scene_paths


def quat_to_rot_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert input quaternion to rotation matrix.

    Args:
        quat (np.ndarray): Input quaternion (w, x, y, z).

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    q = np.array(quat, dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < 1e-10:
        return np.identity(3)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array(
        (
            (1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]),
            (q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]),
            (q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]),
        ),
        dtype=np.float64,
    )
    
    
def quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """
    Quaternion [x, y, z, w] -> 3x3 Rotationsmatrix
    """
    rotmat = R.from_quat(quat).as_matrix()
    return rotmat