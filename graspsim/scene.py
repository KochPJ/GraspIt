from __future__ import annotations

from typing import Any, Dict, List, Optional, ClassVar, Union, Tuple
from dataclasses import dataclass

from curobo.types.math import Pose
from curobo.types.base import TensorDeviceType

from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.prims import create_prim, define_prim, delete_prim
from omni.isaac.core.utils.stage import add_reference_to_stage, open_stage

from geometry import DynamicObject
from utils import get_table_usd
from matrix import T_Matrix

import torch
import numpy as np
import h5py


@dataclass
class Scene(XFormPrim):
    prim_path: str = None    
    name: str = None
    
    # List of Objects   
    objects: Optional[List[DynamicObject]] = None
    table: Optional[DynamicObject] = None
    names: Optional[List[str]] = None

    grasp_path: str = None
    work_path: str = None

    _grasps: List[Pose] = None
    _tensor_args: TensorDeviceType = None

    position: Optional[Union[np.ndarray, torch.Tensor]] = None
    orientation: Optional[Union[np.ndarray, torch.Tensor]] = None

    def __post_init__(self) -> None:
        """
        """

        super().__init__(prim_path=self.prim_path, name=self.name, position=self.position, orientation=self.orientation)

        self._grasps = []
        self._tensor_args = TensorDeviceType()

        if self.grasp_path is not None:
            with h5py.File(self.grasp_path, "r") as f:
                for name in self.names:
                    grasps = np.array(f[name][()])
                    position = []
                    rotation = []

                    for i, (g0, g1) in enumerate(grasps):
                        t0, t1 = T_Matrix(g0), T_Matrix(g1)
                        tran0 = t0.get_translation()
                        tran1 = t1.get_translation()
                        position.append(tran0)
                        position.append(tran1)
                        rotation.append(t0.get_matrix())
                        rotation.append(t1.get_matrix())

                    if len(position) == 0:
                        position.append(np.zeros((3)))
                        rotation.append(np.zeros((3, 3)))
                        
                    pose = Pose(position=self._tensor_args.to_device(position),
                                rotation=self._tensor_args.to_device(rotation),
                                name=name, 
                                normalize_rotation=False)
                    self._grasps.append(pose)

            del grasps
            del position
            del rotation
            del pose

        self._scene_permeability = False
        
        return

    @staticmethod
    def from_dict(data_dict: Dict[str, Any], root_path: str = "/World/Workstation") -> Scene:
        """
        """
        work_path = root_path + "_0"
        scene_path = "/scene"

        _ = define_prim(work_path)
        _ = create_prim(prim_path=work_path + scene_path)

        names = []
        grasp_path = ""
        
        table_usd = get_table_usd()
        # table_usd = data_dict["object0"]["file_path"]
        _ = create_prim(prim_path=work_path + scene_path + "/" + "table", usd_path=table_usd, scale=[0.01])

        # load yaml
        for key in data_dict.keys():
            if key == "grasp_path":
                grasp_path = data_dict[key]
            elif "object" in key and key != "table":
                file_path_usd = data_dict[key]["file_path"]
                name = data_dict[key]['name']
                orientation = data_dict[key]['orientation']
                position = data_dict[key]['position']
                scale = data_dict[key]['scale']

                _ = create_prim(prim_path=work_path + scene_path +"/" + name,
                                usd_path=file_path_usd,  # "file:../" + file_path_usd,
                                orientation=orientation,
                                position=position,
                                scale=scale)
                
                names.append(name)

        if grasp_path == "":
            raise Exception("The \"grasp_path\" is not defined.")
        
        if len(names) == 0:
            raise Exception("There are no objects in the scene.")

        return Scene(
            names=names,
            grasp_path=grasp_path, 
            work_path=work_path,
            prim_path=work_path + scene_path,
            name="scene"
        )
    
    def get_grasps(self, end_effector_offset: Union[list, np.ndarray] = None) -> List[Pose]:
        """
        """
        if end_effector_offset is not None:
            grasps = []
            print("-------Num Grasp per Object--------")
            for i, grasp in enumerate(self._grasps):
                print(f"{i}. ", grasp.position.shape)
                grasps.append(Pose(
                    position=grasp.position.clone() + self._tensor_args.to_device(end_effector_offset),
                    quaternion=grasp.quaternion.clone(),
                    batch=grasp.position.shape[0],
                    normalize_rotation=grasp.normalize_rotation
                    )
                )

            return grasps
        else:
            return self._grasps
    
    @property
    def num_objects(self) -> int:
        """
        """
        return len(self.names)
    
    def get_num_graps(self) -> List[int]:
        """
        """
        return [g.batch for g in self._grasps]
    
    def initialize(self, physics_sim_view=None) -> None:
        """
        """
        super().initialize(physics_sim_view)
        self.table.initialize(physics_sim_view)
        for obj in self.objects:
            obj.initialize(physics_sim_view)

    # TODO: 
    def activate_scene_permeability(self) -> None:
        """
        """
        for obj in self.objects:
            if not obj.get_collision_enabled():
                obj.set_collision_enabled(True)
                # after enabling collision, set the world pose to its initialization pose
                # so that the effects of collision applied in the permeable state do not occur
                obj.set_world_pose(position=obj.get_world_pose()[0], orientation=obj.get_world_pose()[1])
                # rigid_body_physics must be enabled after set_world_pose
                # so that enable_rigid_body_physics works correctly
                obj.enable_rigid_body_physics()

        self.table.set_collision_enabled(True)
        self.table.set_world_pose(position=self.table.get_world_pose()[0], orientation=self.table.get_world_pose()[1])
        self.table.enable_rigid_body_physics()

        self._scene_permeability = True
    
    # TODO: 
    def deactivate_scene_permeability(self, obj_idx: List = None) -> None:
        """
        """
        if obj_idx is None:
            obj_idx = list(range(self.num_objects))

        for i in obj_idx:
            obj = self.objects[i]
            obj.set_collision_enabled(False)
            obj.disable_rigid_body_physics()

        self.table.set_collision_enabled(False)
        self.table.disable_rigid_body_physics()
            
        self._scene_permeability = False

    def is_permeable(self) -> bool:
        """
        """
        return self._scene_permeability
    
    def get_init_pose(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        """
        return [obj.get_local_pose() for obj in self.objects]
    
    def shift(self, offset: Optional[np.ndarray] = None, obj_idx: List = None) -> None:
        """
        """
        if offset is None:
            offset = np.array([0.0, 0.0, 0.0])

        if obj_idx is None:
            pos = self.get_world_pose()[0]
            new_pos = pos+offset
            self.set_world_pose(new_pos)
            self.set_default_state(new_pos)

        else:
            pos = self.table.get_world_pose()[0]
            new_pos = pos+offset
            self.table.set_world_pose(new_pos)
            self.table.set_default_state(new_pos)

            for i in obj_idx:
                obj = self.objects[i]
                pos = obj.get_world_pose()[0]
                new_pos = pos+offset
                obj.set_world_pose(new_pos)
                obj.set_default_state(new_pos)

    def reset(self):
        """
        """

        self.post_reset()
        self.table.post_reset()
        for obj in self.objects:
            obj.post_reset()

    def delete(self):
        """
        """
        delete_prim(self.table.prim_path)
        for obj in self.objects:
            delete_prim(obj.prim_path)

        delete_prim(self.prim_path)
        self._grasps = []

    

"""
from omni.isaac.core.prims.xform_prim import XFormPrim
import numpy as np


ff = open("/home/karaadem/git/file.txt", "w")
for i in range(26):
	if i == 0:
		path = "/Root/Sphere"
	elif i < 10:
		path = "/Root/Sphere_0" + str(i)
	else:
		path = "/Root/Sphere_" + str(i)
	
	f = XFormPrim(prim_path=path)
	a = np.array(f.get_world_pose()[0])
	print("- \"center\""+": ", np.array2string(a, separator=', '))
	print("  \"radius\": ", 0.002)
	t = "- \"center\""+": " + str(np.array2string(a, separator=', '))
	t1 = "  \"radius\": " + str(0.002)
	ff.write(t + "\n")
	ff.write(t1 + "\n")
ff.close()
"""
