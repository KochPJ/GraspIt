from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import trimesh

from utlis import usd2obj, quat_to_rot_matrix

# Constants
DEFAULT_SCALE: tuple[float, float, float] = (0.001, 0.001, 0.001)
POINT_CLOUD_COLOR = [0, 0, 0, 255]


class Object:
    """Represents a 3D object with an associated mesh."""

    def __init__(
        self,
        name: str = "object",
        file_path: str = "",
        orientation: Optional[Sequence[float]] = None,
        position: Optional[Sequence[float]] = None,
        matrix: Optional[np.ndarray] = None,
        scale: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Initializes a 3D object from a mesh file.

        Args:
            name: Identifier for the object.
            file_path: Path to the mesh file (with or without extension).
            orientation: Orientation as quaternion (w, x, y, z).
            position: Translation as (x, y, z).
            matrix: Full 4x4 transformation matrix (takes precedence over position/orientation).
            scale: Scale factors (x, y, z). Default: DEFAULT_SCALE.
        """
        self.name = name
        self.mesh = self._load_mesh(file_path)

        self.rescale(scale if scale is not None else DEFAULT_SCALE)

        if matrix is not None or (position is not None and orientation is not None):
            self.transform(
                position=position,
                orientation=orientation,
                matrix=np.asarray(matrix) if matrix is not None else None,
            )

    @staticmethod
    def _load_mesh(file_path: str) -> trimesh.Trimesh:
        """
        Loads a mesh from OBJ, STL, or USD.

        Args:
            file_path: Path to the mesh file (extension is checked automatically).

        Returns:
            Loaded trimesh object.

        Raises:
            FileNotFoundError: If no supported file is found.
        """
        root = Path(file_path).with_suffix("")

        obj_path = root.with_suffix(".obj")
        stl_path = root.with_suffix(".stl")
        usd_path = root.with_suffix(".usd")

        if obj_path.exists():
            mesh_path = obj_path
        elif stl_path.exists():
            mesh_path = stl_path
        elif usd_path.exists():
            # Convert USD to OBJ and load the generated OBJ
            usd2obj(str(usd_path), str(obj_path))
            mesh_path = obj_path
        else:
            raise FileNotFoundError(
                f"No mesh file found: {obj_path}, {stl_path}, {usd_path}"
            )

        mesh = trimesh.load(mesh_path)

        # trimesh.load may return a Scene – merge into a single mesh
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        return mesh

    @classmethod
    def from_mesh(
        cls,
        mesh: Union[trimesh.Trimesh, trimesh.PointCloud],
        name: str = "object",
    ) -> "Object":
        """
        Creates an Object directly from an existing mesh or PointCloud.

        PointClouds are converted into an empty Trimesh with vertices.

        Args:
            mesh: Existing Trimesh or PointCloud.
            name: Identifier for the object.

        Returns:
            New Object with the provided mesh.
        """
        if isinstance(mesh, trimesh.PointCloud):
            mesh = trimesh.Trimesh(vertices=mesh.vertices)
            mesh.visual.vertex_colors = np.full(
                (len(mesh.vertices), 4), POINT_CLOUD_COLOR, dtype=np.uint8
            )

        obj = cls.__new__(cls)
        obj.name = name
        obj.mesh = mesh
        return obj

    def rescale(self, scale: Sequence[float]) -> None:
        """
        Scales the mesh.

        Args:
            scale: Scale factors as (x, y, z).
        """
        self.mesh.apply_scale(scale)

    def transform(
        self,
        position: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
        matrix: Optional[np.ndarray] = None,
    ) -> None:
        """
        Applies a transformation to the mesh.

        Either a 4x4 transformation matrix is provided directly,
        or it is constructed from position and orientation (quaternion).

        Args:
            position: Translation as (x, y, z).
            orientation: Orientation as quaternion (w, x, y, z).
            matrix: Full 4x4 transformation matrix (takes precedence).

        Raises:
            ValueError: If neither matrix nor position+orientation are provided.
        """
        if matrix is None:
            if position is None or orientation is None:
                raise ValueError(
                    "Either 'matrix' or 'position' + 'orientation' must be provided."
                )
            matrix = np.eye(4)
            matrix[:3, :3] = quat_to_rot_matrix(orientation)
            matrix[:3, 3] = np.asarray(position)

        self.mesh.apply_transform(matrix)