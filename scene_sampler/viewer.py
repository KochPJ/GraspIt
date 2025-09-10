from omni.isaac.franka import Franka
from omni.isaac.sensor import Camera
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction

from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.core.objects.cuboid import DynamicCuboid, FixedCuboid

import omni.usd
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.bounds import compute_aabb, create_bbox_cache
from omni.isaac.core.utils.rotations import euler_angles_to_quat

from omni.isaac.cloner import GridCloner
from utils import get_current_stage

from pxr import Usd, UsdGeom, Gf
import omni.isaac.core.utils.prims as prims_utils
import omni.replicator.core as rep
from omni.usd import get_context
from omni.isaac.dynamic_control import _dynamic_control as dc
from geometry import DynamicObject
from scene import Scene
import os


import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Union

import omni.isaac.core.utils.bounds as bounds_utils
from pxr import Usd
from typing import Dict


class Viewer(object):
    def __init__(self, world: omni.isaac.core.World, scene: Scene, root_path: str, mode: str) -> None:
        """Constructor for synthetic scene viewer

        Args:
            world (omni.isaac.code.World): Isaac Sim World for simulation
            scene (Scene): Isaac Sim Scene for synthetic data generation
            root_path (str): Root path for scene XForms in World
            mode (str): Mode for asset generation, *random* for random position, *linear* for stacking and 
                        non statistical distribution
        """
        
        self.world = world
        self.stage = get_context().get_stage()
        self.names = set(scene.names)
        self.base_scene = scene
        self.scenes: List[Scene] = [scene]
        self.check_flag: bool = False  
        self.step_time: float = 0.0
        self.plane_z: float = 0.0
        self.max_x: float = 0.0
        self.min_x: float = 0.0
        self.max_y: float = 0.0
        self.min_y: float = 0.0        
        self.mode: str = mode
        self.objects = set()

        self.lookup: Dict = {}
        
        self.setup_physics_context()
        if mode == "custom":
            self.setup_custom()
        else:
            self.setup_scene()

        print("==========================================")
        print("Set up Viewer for {}-mode tabletop with {} objects".format(self.mode, len(self.names)))
        print("==========================================")

        self.world.remove_physics_callback("physics_steps")
        self.world.add_physics_callback("physics_steps",callback_fn=self.physics_step)
    
    def setup_physics_context(
            self, 
            physics_dt: float = 1.0 / 60.0,
            gpu_dynamics: bool = False,
            stablization: bool = False,
            gravity: float = -9.81
    ) -> None:
        """_summary_

        Args:
            physics_dt (float, optional): Step size for physics simulation. Defaults to 1.0/60.0.
            gpu_dynamics (bool, optional): GPU optimization, disabled becaus it slows down simulation. Defaults to False.
            stablization (bool, optional): _description_. Defaults to False.
            gravity (float, optional): Gravity parameter, default to Earth. Defaults to -9.81.
        """
        physicsContext = self.world.get_physics_context()
        physicsContext.enable_ccd(True)
        physicsContext.enable_gpu_dynamics(gpu_dynamics)
        physicsContext.enable_stablization(stablization)
        physicsContext.set_physics_dt(physics_dt)
        gravity = gravity * (self.mode != "custom")
        physicsContext.set_gravity(gravity)
        return


    def setup_scene(self) -> None:
        """Setup for synthetic scene, population, randomization and scaling of assets
        """
        self.world.scene.add_default_ground_plane()

        stage = get_current_stage()
        unit = UsdGeom.LinearUnits.meters
        UsdGeom.SetStageMetersPerUnit(stage, unit)
        
        assert UsdGeom.GetStageMetersPerUnit(stage) == unit

        self.objects = []
        table = DynamicObject(prim_path=self.scenes[0].work_path + "/" + "table",
                              name="table_0",
                              collision=True,
                              rigid_body_physics=False,
                              approximation=None,
                              use_visual_material=False,
                              use_physics_material=False,
                              scale=[0.01, 0.01, 0.01])
        self.world.scene.add(table)
        self.lookup["table"] = [0.01, 0.01, 0.01]

        bounding_box = table.compute_bb()
        table_scale = np.array(bounding_box[3:]) - np.array(bounding_box[:3])
        self.min_x, self.max_x = bounding_box[0], bounding_box[3]
        self.min_y, self.max_y = bounding_box[1], bounding_box[4]
        self.plane_z = bounding_box[-1]
        print(self.plane_z)
        collision_position = table.get_world_pose()[0]
        collision_position[2] = bounding_box[-1] - table_scale[2]

        for name in self.scenes[0].names:
            if self.mode == "random":
                translation = np.array([np.random.uniform(self.min_x - self.min_x * 0.3, self.max_x - self.max_x * 0.3),
                                        np.random.uniform(self.min_y -self.min_y * 0.3, self.max_y - self.max_y * 0.3),
                                        np.random.uniform(self.plane_z + 0.5, self.plane_z + 2.5)])
                orientation = np.array([np.random.random(),
                                        np.random.uniform(0, 360),
                                        np.random.uniform(0, 360),
                                        np.random.uniform(0, 360)])
            elif self.mode == "linear":
                translation = np.array([0.0, 0.0, self.plane_z + 0.5])
                
            obj_bounding_box = omni.usd.get_context().compute_path_world_bounding_box(self.scenes[0].work_path + "/" + name)
            obj_scale = np.array(obj_bounding_box[1]) - np.array(obj_bounding_box[0])

            factors = [obj_scale[0] / 0.08, obj_scale[1] / 0.08, obj_scale[2] / 0.08]
            _max = max(factors)
            print(factors, _max)
            ################################
            # scaling objects to fit on table and have sizes in 'reasonable' ranges, e.g not beeing to small or to large
            ################################
            if _max in [0.1, 1]:
                _object = DynamicObject(prim_path=self.scenes[0].work_path + "/" + name,
                                       name=name,
                                       #color=np.random.rand(1, 3),
                                       translation=translation,
                                       orientation=orientation,
                                       scale = [1.0, 1.0, 1.0])
                self.lookup[name] = [1.0, 1.0, 1.0]
            elif _max > 1:
                _object = DynamicObject(prim_path=self.scenes[0].work_path + "/" + name,
                                       name=name,
                                       #color=np.random.rand(1, 3),
                                       translation=translation,
                                       orientation=orientation,
                                       scale = [1 / _max, 1 / _max, 1 / _max])
                self.lookup[name] = [1 / _max, 1 / _max, 1 / _max]
            else:
                _object = DynamicObject(prim_path=self.scenes[0].work_path + "/" + name,
                                   name=name,
                                   #color=np.random.rand(1, 3),
                                   translation=translation,
                                   orientation=orientation,
                                   scale = [5.0, 5.0, 5.0])
                self.lookup[name]= [5.0, 5.0, 5.0]

            self.objects.append(_object)
            self.world.scene.add(_object)
        return
    
    def save_scene(self, count: int, yaml_path: str):
        """generating .yaml file to save scene and enable later reloading

        Args:
            count (int): index of current scene
            yaml_path (str): path of .yaml file
        """
        import yaml

        print("Saving Scene")
        print("==========================================")
        scene_dict = {}

        scene_dict["table"] = {}
        scene_dict["table"]["name"] = "table"
        scene_dict["table"]["file_path"] = "/share/assets/table/table.obj"
        scene_dict["table"]["position"] = np.array([0.0, 0.0, 0.0]).tolist()
        scene_dict["table"]["orientation"] = np.array([0.0, 0.0, 0.0, 0.0]).tolist()
        scene_dict["table"]["scale"] = np.array([0.01, 0.01, 0.01]).tolist()
        print(scene_dict)
    
        for index, _object in enumerate(self.objects):
            name = _object.name
            number = name.strip("object")
            print(index, _object, name)
            file_path = "/share/assets/{}/{}.usd".format(number, number)
            scene_dict["object{}".format(index)] = {}
            file_path = "/share/assets/{}/{}.usd".format(number, number)
            translation, orientation = _object.get_world_pose()
            scene_dict["object{}".format(index)]["name"] = name
            scene_dict["object{}".format(index)]["file_path"] = file_path
            scene_dict["object{}".format(index)]["position"] = translation.tolist()
            scene_dict["object{}".format(index)]["orientation"] = orientation.tolist()
            scene_dict["object{}".format(index)]["scale"] = np.array(self.lookup[name]).tolist()

    

        os.makedirs("out/scenes", exist_ok=True)
        with open(f"{yaml_path}/{count}.yaml", "w") as f:
            yaml.dump(scene_dict, f)

    def physics_step(self, step_size: int) -> None:
        """Step function for physics simulation, allows for removal of unwanted objects

        Args:
            step_size (int): _description_
        """
        if not self.check_flag:
            if self.step_time > 8:
                self.disable_fallen()
                self.check_flag = True
    
    def disable_fallen(self) -> None:
        """Removes objects fallen from tabletop from scene
        """
        count = 0
        for index, object in enumerate(self.objects):
            
            pose = object.get_world_pose()
            if pose[0][-1] < self.plane_z * 0.5:
                prims_utils.delete_prim(object.prim_path)
                count += 1
                self.objects.remove(object)
            elif pose[0][0] < self.min_x or pose[0][0] > self.max_x:
                prims_utils.delete_prim(object.prim_path)
                count += 1
                self.objects.remove(object)
            elif pose[0][1] < self.min_y or pose[0][1] > self.max_y:
                prims_utils.delete_prim(object.prim_path)
                count += 1
                self.objects.remove(object)
        print("Deleted {} objects for exceeding table limits".format(count))
        print("==========================================")

    def post_reset(self) -> None:
        """reset helper for initial reset of simulation
        """
        #self.base_scene.table.post_reset()
        if self.base_scene.objects is not None:
            for object in self.base_scene.objects:
                object.post_reset()
        return