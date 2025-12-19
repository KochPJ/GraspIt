from __future__ import annotations

from object import Object
import trimesh

from typing import Any, Dict, List, Optional, Sequence
from dataclasses import dataclass

from utlis import create_plane
import io
from PIL import Image
import numpy as np
from pyglet import gl



@dataclass
class Scene(Sequence):

    #: List of Objects.
    object: Optional[List[Object]] = None
    table: Optional[Object] = None
    

    def __post_init__(self) -> None:
        """
        """
        self.scene_viewer = trimesh.Scene()
        self.collision_manager = trimesh.collision.CollisionManager()
        
        meshes = []
        for obj in self.object:
            obj.mesh.visual.kind == 'face'
            obj.mesh.visual.face_colors = np.array(np.random.randint(0, 256, size=3).tolist() + [255])
            meshes.append(obj.mesh)
            self.collision_manager.add_object(obj.name, obj.mesh)

        self.scene_mesh = trimesh.util.concatenate(meshes)
        self.scene_mesh.visual.kind == 'face'
        # self.scene_mesh.visual.face_colors = trimesh.visual.random_color()

        table_bounds = self.table.mesh.bounding_box.bounds
        scene_bounds = self.scene_mesh.bounding_box.bounds
        self.table_height = table_bounds[1][2]
        print(self.table_height)

        plane = create_plane(table_bounds, scene_bounds)
        
        self.collision_manager.add_object('plane', plane)
        self.scene_mesh = trimesh.util.concatenate([self.scene_mesh, plane])
        self.scene_mesh.visual.kind == 'face'

        self.scene_viewer.add_geometry(self.scene_mesh)
        #self.scene_viewer.add_geometry(self.table.mesh)

    def add(self, obj: Object) -> str:
        """
        """
        name = self.scene_viewer.add_geometry(obj)
        return name
    
    def delete(self, name: str) -> None:
        """
        """
        self.scene_viewer.delete_geometry(name)
        self.scene_viewer.graph.transforms.remove_node(name)

    def save_image(self, resolution):
        """
        """
        window_conf = gl.Config(double_buffer=True, depth_size=32)
        data = self.scene_viewer.save_image(resolution=[1920*2, 1080*2], window_conf=window_conf)
        image = Image.open(io.BytesIO(data))
        image.show()
        image.save("data/scene1.png")

    def in_collision_with(self, marker):
        """
        """
        in_collision = self.collision_manager.in_collision_single(marker,
                                                                  return_names=False,
                                                                  return_data=False)

        return in_collision
    
    def min_distance_with(self, marker):
        min_distance = self.collision_manager.min_distance_single(marker,
                                                                  return_name=False,
                                                                  return_data=False)

        return min_distance

    def show(self):
        self.scene_viewer.show()

    def __len__(self):
        return len(self.object)

    def __getitem__(self, idx):
        return self.object[idx]

    @staticmethod
    def from_dict(data_dict: Dict[str, Any]) -> Scene:
        """
        """
        object = []
        table = None

        # load yaml:
        for key in data_dict.keys():
            if key == "table":
                table = Object(name="table", 
                               file_path=data_dict[key]["file_path"], 
                               scale=data_dict[key]["scale"])

            elif "object" in key:
                print(key)
                object.append(Object(**data_dict[key]))
        
        return Scene(
            object=object,
            table=table
        )
