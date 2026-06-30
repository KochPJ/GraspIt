"""Module for managing a 3D scene with objects and collision detection."""

from __future__ import annotations

import copy
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import trimesh
from PIL import Image
from pyglet import gl

from object import Object
from utlis import create_plane


@dataclass
class SceneItem:
    """An element added to the scene, associated with a parent object."""

    name: str
    geometry: any  # trimesh.Trimesh
    parent_id: int


@dataclass
class Scene:
    """3D scene with objects, collision manager, and visualization.

    Supports sequence access to the contained objects.
    """

    objects: list[Object] = field(default_factory=list)
    table: Optional[Object] = None

    def __post_init__(self) -> None:
        if not self.objects:
            raise ValueError("The scene requires at least one object.")

        self.collision_manager = trimesh.collision.CollisionManager()
        self.items: list[SceneItem] = []

        meshes: list[trimesh.Trimesh] = []

        for obj in self.objects:
            color = np.array(
                [*np.random.randint(0, 256, size=3), 255], dtype=np.uint8
            )
            obj.mesh.visual.face_colors = color
            meshes.append(obj.mesh)
            self.collision_manager.add_object(obj.name, obj.mesh)

        self.scene_mesh = trimesh.util.concatenate(meshes)

        scene_bounds = self.scene_mesh.bounding_box.bounds
        self.table_height: float = scene_bounds[0][2]

        self.plane = create_plane(scene_bounds)
        self.collision_manager.add_object("plane", self.plane)

        self.scene_mesh = trimesh.util.concatenate([self.scene_mesh, self.plane])

    def __len__(self) -> int:
        return len(self.objects)

    def __getitem__(self, idx: int) -> Object:
        return self.objects[idx]

    def copy(self) -> Scene:
        """Creates a deep copy of the scene."""
        return copy.deepcopy(self)

    def add(
        self,
        geom: Union[Object, trimesh.Trimesh, trimesh.PointCloud],
        parent_id: int,
        name: Optional[str] = None,
    ) -> None:
        """Adds a geometry element to the scene.

        Args:
            geom: Geometry as Object, Trimesh, or PointCloud.
            parent_id: Index of the parent object.
            name: Optional name (auto-generated if None).
        """
        obj = self._to_object(geom, name=name)
        self.items.append(
            SceneItem(name=obj.name, geometry=obj.mesh, parent_id=parent_id)
        )

    def delete(self, parent_id: int) -> None:
        """Removes all items associated with the specified parent object."""
        self.items = [item for item in self.items if item.parent_id != parent_id]

    def reset(self) -> None:
        """Removes all added items from the scene."""
        self.items.clear()

    def in_collision_with(self, mesh: trimesh.Trimesh) -> bool:
        """Checks whether the given mesh collides with the scene."""
        return self.collision_manager.in_collision_single(
            mesh, return_names=False, return_data=False
        )

    def min_distance_to(self, mesh: trimesh.Trimesh) -> float:
        """Returns the minimum distance from the mesh to the scene."""
        return self.collision_manager.min_distance_single(
            mesh, return_name=False, return_data=False
        )

    def show(self, obj_ids: Optional[list[int]] = None) -> None:
        """Displays the scene in an interactive window."""
        self._create_viewer_scene(obj_ids).show()

    def save_image(
        self,
        output_path: Union[str, Path] = "data/scene.png",
        resolution: tuple[int, int] = (1920, 1080),
        obj_ids: Optional[list[int]] = None,
    ) -> Image.Image:
        """Renders the scene and saves it as an image.

        Args:
            output_path: Destination file path for the image.
            resolution: Resolution as (width, height).
            obj_ids: Optional list of object indices to filter.

        Returns:
            The rendered PIL image.
        """
        scene = self._create_viewer_scene(obj_ids)

        window_conf = gl.Config(double_buffer=True, depth_size=32)
        data = scene.save_image(resolution=resolution, window_conf=window_conf)

        image = Image.open(io.BytesIO(data))

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)

        return image

    @staticmethod
    def from_dict(data_dict: dict[str, Any]) -> Scene:
        """Creates a scene from a dictionary (e.g., YAML data).

        Expects keys like 'table' and 'object_*' with the respective parameters.
        """
        objects: list[Object] = []
        table: Optional[Object] = None

        for key, value in data_dict.items():
            if key == "table":
                table = Object(
                    name="table",
                    file_path=value["file_path"],
                    scale=value["scale"],
                )
            elif key.startswith("object"):
                objects.append(Object(**value))

        return Scene(objects=objects, table=table)

    def _to_object(self, geom: Any, name: Optional[str] = None) -> Object:
        """Converts various geometry types into an Object."""
        if isinstance(geom, Object):
            return geom

        if isinstance(geom, (trimesh.Trimesh, trimesh.PointCloud, trimesh.path.path.Path)):
            if name is None:
                name = f"item_{len(self.items)}"
            return Object.from_mesh(geom, name)

        raise TypeError(f"Unsupported type: {type(geom).__name__}")

    def _create_viewer_scene(
        self, obj_ids: Optional[list[int]] = None
    ) -> trimesh.Scene:
        """Builds a trimesh.Scene for visualization/export.

        Args:
            obj_ids: Optional indices; if provided, only include these objects.

        Raises:
            IndexError: If invalid object indices are provided.
        """
        if obj_ids is not None:
            invalid = [i for i in obj_ids if not 0 <= i < len(self.objects)]
            if invalid:
                raise IndexError(f"Invalid object indices: {invalid}")
            obj_id_set = set(obj_ids)
        else:
            obj_id_set = None

        scene = trimesh.Scene()
        for i, obj in enumerate(self.objects):
            if obj_id_set is not None and i not in obj_id_set:
                continue
            scene.add_geometry(obj.mesh, node_name=obj.name)

            for item in self.items:
                if item.parent_id == i:
                    scene.add_geometry(item.geometry, node_name=item.name)

        if self.plane is not None:
            scene.add_geometry(self.plane, node_name="plane")

        return scene
    