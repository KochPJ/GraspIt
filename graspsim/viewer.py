from omni.isaac.franka import Franka
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction

from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.core.objects.cuboid import DynamicCuboid, FixedCuboid
from omni.isaac.sensor.scripts.effort_sensor import EffortSensor

from omni.isaac.core.objects import cuboid, sphere
import omni.usd

from omni.isaac.core.utils.prims import get_prim_at_path, delete_prim
from omni.isaac.core.utils.rotations import euler_angles_to_quat, rot_matrix_to_quat, matrix_to_euler_angles

from omni.isaac.core.utils.viewports import set_camera_view

from omni.isaac.cloner import GridCloner
from omni.isaac.core import World

from geometry import DynamicObject

from controller2 import RobotController

from scene import Scene
from environment import Environment, EnvironmentScheduler
from observer import RobotObserver

from constants import *
import os

from scipy.spatial.transform import Rotation as R
import numpy as np
import json
from typing import List, Dict
from curobo.util_file import load_yaml

from pxr import Usd


class SimViewer(object):
    """In this class you will find all the code of the simulation behavior.

    Args:
        world: Isaac Sim World object
    """

    def __init__(self, world: World, scene_paths: List, root_path: str) -> None:
        self.world = world
        self.scene_paths = scene_paths

        self.scene_roots = []
        for path in scene_paths:
            root, filename = os.path.split(path)
            self.scene_roots.append(root)

        self.scene_path_idx = 0
        self.scene_path_idx = 0
        self.root_path = root_path

        self.n_env = MAX_ENV

        # Slip Dataset
        self.slip = {}
        self.grasp = {}
        self.grasp_single = {}

        # Phase 0
        self.cmd_plan = None
        self.succ_idx = None
        self.cmd_idx_list = [0]*MAX_ENV
        self._settle_time = [0]*MAX_ENV
        self._t_close = [0]*MAX_ENV

        # Move Up Actions
        self._action_sequence = [[] for _ in range(self.n_env)]
        self._action_sequence_index = [0]*self.n_env

        # Linear oscillation Actions
        self._linear_action_sequence = [[] for _ in range(self.n_env)]
        self._linear_action_sequence_index = [0]*self.n_env

        # Pendulum 
        self._period_pendulum_number = [0]*self.n_env
        self._reached_max_flags = [False]*self.n_env
        self._reached_min_flags = [False]*self.n_env

        # TODO: Calculate table height dynamically
        self.table_height = 0.82436067

        # Begin with the first object
        self.obj_id = 0
        self.prev_obj_id = self.obj_id
        self.prev_state = None

        self.n_obj = None

        self._next_scene = False
        self._finished = False

        # Franka joint names
        self.joint_names = [
            "panda_leftfinger",
            "panda_rightfinger",
            "panda_hand",
            "panda_link8",
            "panda_link7",
            "panda_link6",
            "panda_link5",
        ]

        # TODO: in Robot Class
        self.joint_forces : Dict[str, RigidPrimView] = {}

        self.collision_table : FixedCuboid = None
        
        # Grid Cloner
        self.cloner = GridCloner(spacing=SPACING, num_per_row=NUM_PER_ROW)
        self.target_paths = self.cloner.generate_paths(root_path=root_path, num_paths=self.n_env)
        self.base_path = self.target_paths[0]
        self.base_robor_path = self.base_path + "/franka"
        self.work_positions, self.work_orientations = self.cloner.get_clone_transforms(self.n_env)
        self.base_work_position = self.work_positions[0]
        self.base_work_orientation = self.work_orientations[0]

        print("------Work Positions-------")
        for wp in self.work_positions:
            print(wp)

        self.envs : List[Environment] = []

        self.setup_physics_context()
        self.setup_world()
        self.config_robot()

        #----Calculate an individual robot position for each object----#
        # TODO: Calculate table height dynamically
        self.robot_positions = self.calculate_robot_positions(self.envs[0].scene, INIT_ROBOTER_POS, self.table_height)
        self.set_roboter_position(self.robot_positions[self.obj_id])

        print("------Robot Positions-------")
        for rp in self.robot_positions:
            print(rp)

        self.robot_orientations = [env.robot.get_local_pose()[1] for env in self.envs]

        self.batch = 25
        self.controller = RobotController(
                                    name="collision_free_trajectory_generator",
                                    articulation=self.envs[0].articualtion,
                                    robot_positions=self.robot_positions,
                                    robot_orientations=self.robot_orientations,
                                    offset_position=np.array(self.base_work_position),
                                    batch=self.batch,
                                    warmup=True
                           )
        self.warmup = False
        
        self.robot_observer = RobotObserver(envs=self.envs,
                                    work_positions=self.work_positions,
                                    robot_positions=self.robot_positions,
                                    controller=self.controller,
                                    leftfinger=self.joint_forces["panda_leftfinger"],
                                    rightfinger=self.joint_forces["panda_rightfinger"]
                              )
        
        self.environment_scheduler = EnvironmentScheduler(envs=self.envs, start_id=self.obj_id)

        # Add physics Step.
        # will be called before each physics step: world.step().
        self.world.add_physics_callback("physics_steps", callback_fn=self.physics_step)
        self.world.scene.enable_bounding_boxes_computations()

        set_camera_view(eye=[2, 0, 1], target=[0.00, 0.00, 0.00], camera_prim_path="/OmniverseKit_Persp")

        self.load_trajectorys = True

        return

    def update_obstacle_world(self) -> None:
        """
        """
        obstacles = self.controller.get_obstacles(only_paths=[self.base_path], ignore_paths=[self.base_robor_path])

        print("----------Obstacles----------")
        for obj in obstacles.objects:
            print('------------------------------------------------------------')
            print("Name: ", obj.name)
            pose = obj.pose
            print("Position: ", pose[:3])
            print("Orientation: ", pose[3:])

        self.controller.update_world(obstacles)

    def setup_physics_context(
        self,
        physics_dt: float = 1.0 / 60.0,
        gpu_dynamics: bool = False,
        stablization: bool = False,
        gravity: float = -9.81,
    ) -> None:
        """Set desired physics Context options
        """
        physicsContext = self.world.get_physics_context()
        physicsContext.enable_gpu_dynamics(gpu_dynamics)  # bloß nicht enablen, ist langsam
        physicsContext.enable_stablization(stablization)
        physicsContext.set_physics_dt(physics_dt)
        physicsContext.set_gravity(gravity)

        return

    def setup_world(self) -> None:
        """
        NOTE:

            If the views has been added to the world scene (e.g., ``world.scene.add(prims)``),
            it will be automatically initialized (i.e., ``prim.initialize()``)
            when the world is reset (e.g., ``world.reset()``).

        WARNING:

            ``prim.initialize()`` needs to be called before interacting with any other class method.
        """

        base_scene = Scene.from_dict(load_yaml(self.scene_paths[self.scene_path_idx]), self.root_path)
        print(self.scene_paths[self.scene_path_idx])

        # Add base robot and articulation to the base scene
        robot_orientation = euler_angles_to_quat(np.array([INIT_ROBOTER_ORI_X,
                                                           INIT_ROBOTER_ORI_Y,
                                                           INIT_ROBOTER_ORI_Z]), degrees=True)
        base_robot = Franka(prim_path=self.base_robor_path,
                            name="franka_0",
                            orientation=robot_orientation)
        
        base_articulation = Articulation(prim_path=self.base_robor_path,
                                         name="articulation_0",
                                         enable_dof_force_sensors=True)

        # add ground plane
        name = "default_ground_plane"
        if not self.world.scene.object_exists(name=name):
            self.world.scene.add_default_ground_plane(name=name)

        scene_path = "/scene"
        # Cloning the base scene based on the number of objects in the base scene
        _ = self.cloner.clone(source_prim_path=self.target_paths[0],
                              prim_paths=self.target_paths,
                              copy_from_source=True)
        
        # Setup base scene
        objects = []
        table = DynamicObject(prim_path=self.target_paths[0] + scene_path + "/" + "table",
                              name="table_0",
                              collision=True,
                              rigid_body_physics=False,
                              approximation=None,
                              use_visual_material=False,
                              use_physics_material=False)
        
        for name in base_scene.names:
            objects.append(DynamicObject(prim_path=self.target_paths[0] + scene_path + "/" + name,
                           name=name + "_0",
                           color=np.random.rand(1, 3)))
            
        base_scene.objects = objects
        base_scene.table = table

        self.envs.append(Environment(prim_path=self.target_paths[0],
                                     name="environment_0",
                                     scene=base_scene,
                                     robot=base_robot,
                                     robot_articualtion=base_articulation))

        # create collision table only for the base scene
        bb = table.compute_bb()
        scale = np.array(bb[3:]) - np.array(bb[:3])
        scale[2] = 0.01
        collision_position = table.get_world_pose()[0]
        collision_position[2] = bb[-1] - scale[2]

        self.collision_table = FixedCuboid(prim_path=self.target_paths[0] + scene_path + "/" + "collision_table",
                                      name="collision_table",
                                      position=collision_position, 
                                      scale=scale, 
                                      visible=True)
        self.world.scene.add(self.collision_table)

        # Setup clones
        for i, path in enumerate(self.target_paths[1:]):
            objects = []
            table = DynamicObject(prim_path=path + scene_path + "/" + "table",
                                  name="table_" + str(i+1),
                                  collision=True,
                                  rigid_body_physics=False,
                                  approximation=None,
                                  use_visual_material=False,
                                  use_physics_material=False)
            
            for name in base_scene.names:
                objects.append(DynamicObject(prim_path=path + scene_path + "/" + name,
                               name=name + "_" + str(i+1), 
                               color=np.random.rand(1, 3)))

            scene = Scene(objects=objects,
                          table=table,
                          work_path=path, 
                          names=base_scene.names, 
                          name="scene", 
                          prim_path=path + scene_path)
            
            robot = Franka(prim_path=path + "/franka", name="franka_" + str(i+1))
            articulation = Articulation(prim_path=path + "/franka",
                                        name="articulation_" + str(i+1),
                                        enable_dof_force_sensors=True)

            self.envs.append(Environment(prim_path=path,
                             name="environment_" + str(i+1),
                             scene=scene,
                             robot=robot,
                             robot_articualtion=articulation))

        for env in self.envs:
            self.world.scene.add(env)

        return
    
    def clean_world(self):
        """
        """
        collision_table = self.world.scene.get_object("collision_table")
        delete_prim(collision_table.prim_path)
        self.world.scene.remove_object(collision_table.name, registry_only=True)

        for joint in self.joint_names:
            self.world.scene.remove_object(self.joint_forces[joint].name, registry_only=True)

        for env in self.envs:
            env.delete()
            self.world.scene.remove_object(env.name, registry_only=True)

        self.envs.clear()
        self.joint_forces.clear()

    def config_robot(self):
        """
        """

        # Contact force Views
        robot_prim_path = self.target_paths[0][:-1] + "*" + "/franka"
        for joint in self.joint_names:
            self.joint_forces[joint] = RigidPrimView(
                prim_paths_expr=robot_prim_path + "/" + joint,
                name=joint + "_view",
                reset_xform_properties=False,
                track_contact_forces=True,
                prepare_contact_sensors=True,
                contact_filter_prim_paths_expr=self.target_paths,
                # max_contact_count=7
            )
            self.world.scene.add(self.joint_forces[joint])

        # Config franka collision fingers, hand, links
        for path in self.target_paths:
            # franka joint prims
            panda_leftfinger_prim = get_prim_at_path(path + "/franka/panda_leftfinger/geometry/panda_leftfinger")
            panda_rightfinger_prim = get_prim_at_path(path + "/franka/panda_rightfinger/geometry/panda_rightfinger")
            panda_hand_prim = get_prim_at_path(path + "/franka/panda_hand/geometry/panda_hand")
            panda_link0 = get_prim_at_path(path + "/franka/panda_link0/geometry/panda_link0")
            panda_link1 = get_prim_at_path(path + "/franka/panda_link1/geometry/panda_link1")
            panda_link7 = get_prim_at_path(path + "/franka/panda_link7/geometry/panda_link7")

            # set hullVertexLimit and maxConvexHulls
            panda_leftfinger_prim.GetAttribute("physxConvexDecompositionCollision:hullVertexLimit").Set(64)
            panda_leftfinger_prim.GetAttribute("physxConvexDecompositionCollision:maxConvexHulls").Set(2048)

            panda_rightfinger_prim.GetAttribute("physxConvexDecompositionCollision:hullVertexLimit").Set(64)
            panda_rightfinger_prim.GetAttribute("physxConvexDecompositionCollision:maxConvexHulls").Set(2048)

            # config ConvexDecompositionCollisionAPI for franka hand
            reg = Usd.SchemaRegistry()
            API = reg.GetAPITypeFromSchemaTypeName("PhysxConvexDecompositionCollisionAPI")
            panda_hand_prim.ApplyAPI(API)
            panda_hand_prim.GetAttribute("physics:approximation").Set("convexDecomposition")
            panda_hand_prim.GetAttribute("physxConvexDecompositionCollision:hullVertexLimit").Set(64)
            panda_hand_prim.GetAttribute("physxConvexDecompositionCollision:maxConvexHulls").Set(2048)
            panda_hand_prim.GetAttribute("physxConvexDecompositionCollision:voxelResolution").Set(620_000)

            # enable shrinkWrap
            panda_leftfinger_prim.GetAttribute('physxConvexDecompositionCollision:shrinkWrap').Set(True)
            panda_rightfinger_prim.GetAttribute('physxConvexDecompositionCollision:shrinkWrap').Set(True)
            panda_hand_prim.GetAttribute('physxConvexDecompositionCollision:shrinkWrap').Set(True)

            # disable collision
            panda_link0.GetAttribute('physics:collisionEnabled').Set(False)
            panda_link1.GetAttribute('physics:collisionEnabled').Set(False)
            panda_link7.GetAttribute('physics:collisionEnabled').Set(False)
            panda_hand_prim.GetAttribute('physics:collisionEnabled').Set(False)

        # dissolve the joint limits
        for path in self.target_paths:
            panda_joint1_prim = get_prim_at_path(path + "/franka/panda_link0/panda_joint1")
            panda_joint2_prim = get_prim_at_path(path + "/franka/panda_link1/panda_joint2")
            panda_joint3_prim = get_prim_at_path(path + "/franka/panda_link2/panda_joint3")
            panda_joint4_prim = get_prim_at_path(path + "/franka/panda_link3/panda_joint4")
            panda_joint5_prim = get_prim_at_path(path + "/franka/panda_link4/panda_joint5")
            panda_joint6_prim = get_prim_at_path(path + "/franka/panda_link5/panda_joint6")
            panda_joint7_prim = get_prim_at_path(path + "/franka/panda_link6/panda_joint7")

            panda_joint1_prim.GetAttribute('physics:lowerLimit').Set(-360.0)
            panda_joint1_prim.GetAttribute('physics:upperLimit').Set(360.0)

            panda_joint2_prim.GetAttribute('physics:lowerLimit').Set(-360.0)
            panda_joint2_prim.GetAttribute('physics:upperLimit').Set(360.0)

            panda_joint3_prim.GetAttribute('physics:lowerLimit').Set(-360.0)
            panda_joint3_prim.GetAttribute('physics:upperLimit').Set(360.0)

            panda_joint4_prim.GetAttribute('physics:lowerLimit').Set(-360.0)
            panda_joint4_prim.GetAttribute('physics:upperLimit').Set(360.0)

            panda_joint5_prim.GetAttribute('physics:lowerLimit').Set(-360.0)
            panda_joint5_prim.GetAttribute('physics:upperLimit').Set(360.0)

            panda_joint6_prim.GetAttribute('physics:upperLimit').Set(270)

            panda_joint7_prim.GetAttribute('physics:lowerLimit').Set(-360.0)
            panda_joint7_prim.GetAttribute('physics:upperLimit').Set(360.0)

            # To stabilize the robot
            API = Usd.SchemaRegistry.GetAPITypeFromSchemaTypeName("PhysxJointAPI")
            panda_joint1_prim.ApplyAPI(API)
            panda_joint1_prim.GetAttribute('physxJoint:armature').Set(0.05)

    def set_roboter_position(self, position: np.ndarray) -> None:
        """Set the local position of the robots
        """
        for env in self.envs:
            env.robot.set_local_pose(position)

    def calculate_robot_positions(self, scene: Scene, init_robot_pos, table_height=0.82436067):
        """
        Berechnet die Roboterpositionen für jedes Objekt basierend auf den gegebenen Parametern.
        """

        # Get the average object poses without the table height
        poses = scene.get_grasps(end_effector_offset=np.array([0, 0, -table_height]))
        mean_pose = [np.mean(pose.position.cpu().numpy(), axis=0) for pose in poses]

        init_roboter_position = np.array([
            init_robot_pos[0],
            init_robot_pos[1],
            init_robot_pos[2] + table_height
        ])
        
        robot_positions = [init_roboter_position + pose for pose in mean_pose]
        return robot_positions

    def reset(self) -> None:
        """Resetting anything in the world should happen here.
        """
        for env in self.envs:
            env.robot.post_reset()
            env.articualtion.post_reset()

        # Slip Dataset
        self.slip = {}
        self.grasp = {}
        self.grasp_single = {}

        # Phase 0
        self.cmd_plan = None
        self.succ_idx = None
        self.cmd_idx_list = [0]*MAX_ENV
        self._settle_time = [0]*MAX_ENV
        self._t_close = [0]*MAX_ENV

        # Move Up Actions
        self._action_sequence = [[] for _ in range(self.n_env)]
        self._action_sequence_index = [0]*self.n_env

        # Linear oscillation Actions
        self._linear_action_sequence = [[] for _ in range(self.n_env)]
        self._linear_action_sequence_index = [0]*self.n_env

        # Pendulum 
        self._period_pendulum_number = [0]*self.n_env
        self._reached_max_flags = [False]*self.n_env
        self._reached_min_flags = [False]*self.n_env

        # Begin with the first object
        self.obj_id = 0
        self.prev_obj_id = self.obj_id
        self.prev_state = None

        self.n_obj = None

        self._next_scene = False

        return

    def reset_environment(self, i: int) -> None:
        """
        """
        self.envs[i].robot.get_articulation_controller().switch_dof_control_mode(dof_index=5, mode="position")
        self.envs[i].robot.get_articulation_controller().switch_control_mode(mode="position")
        self.envs[i].robot.post_reset()
        self.envs[i].articualtion.post_reset()

        self.envs[i].scene.reset()
        self.envs[i].shift_scene(offset=np.array([-40.0, 0.0, 0.0]))
        self.envs[i].scene._scene_permeability = False
        self._settle_time[i] = 0.0
        self._t_close[i] = 0.0

        self._action_sequence[i] = []
        self._action_sequence_index[i] = 0

        self._linear_action_sequence[i] = []
        self._linear_action_sequence_index[i] = 0

        self._period_pendulum_number[i] = 0
        self._reached_max_flags[i] = False
        self._reached_min_flags[i] = False

        self.cmd_idx_list[i] = 0

        if self.envs[i].get_env_state() == Environment.State.TERMINATED:
            self.envs[i].set_env_state(Environment.State.FINISHED)

    def terminate_environment(self, i: int) -> None:
        """
        """
        self.envs[i].robot.set_joint_velocities(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        self._settle_time[i] = MAX_SETTLE_TIME
        self._t_close[i] = MAX_CLOSE_TIME

        target_joints = [None] * self.envs[i].articualtion.get_joint_positions().shape[0]
        self._linear_action_sequence[i] = [ArticulationAction(joint_positions=target_joints)]
        self._linear_action_sequence_index[i] += 1
        self._period_pendulum_number[i] = MAX_PERIOD

    @property
    def get_step_index(self) -> int:
        """
        """
        return self.world.current_time_step_index

    def physics_step(self, step_size):
        """Function runs before every physics frame

        Args:
            step_size: time since last physics step. Depends on physics_dt

        Raises:
            Exception: [description]
        """

        step_index = self.get_step_index

        #! Plan or load trajectorys
        if self.cmd_plan is None:
            self._next_scene = False

            if self.load_trajectorys:
                try:
                    filename = os.path.join(self.scene_roots[self.scene_path_idx], "trajectorys" + ".json")
                    self.cmd_plan, self.succ_idx = self.controller.load_trajectorys(filename=filename)

                    n_graps = [len(g) for g in self.cmd_plan]
                    self.n_obj = len(n_graps)

                    self.environment_scheduler.set_num_grasp(n_graps)
                    self.prev_state = self.environment_scheduler.reset()
                    print("Number of Grasps: ", n_graps)
                    print(self.succ_idx[0])

                    # set the unused envs to IDLE
                    all_envs = list(range(self.n_env))
                    for i in all_envs[n_graps[0]:]:
                        self.envs[i].set_env_state(Environment.State.IDLE)

                except Exception as e:
                    print(f"Fehler beim Laden der Trajektorien: {e}")

            if not self.load_trajectorys or 'e' in locals():
                self.update_obstacle_world()
                print(f"Scene: {len(self.scene_paths)}/{self.scene_path_idx+1}")
                self.cmd_plan, self.succ_idx = self.controller.plan(
                    self.envs[0].scene.get_grasps(end_effector_offset=self.base_work_position)
                )
                filename = os.path.join(self.scene_roots[self.scene_path_idx], "trajectorys" + ".json")
                self.controller.save_trajectorys(self.cmd_plan, self.succ_idx, filename)

                n_graps = [len(g) for g in self.cmd_plan]
                self.n_obj = len(n_graps)

                self.environment_scheduler.set_num_grasp(n_graps)
                self.prev_state = self.environment_scheduler.reset()
                print("Number of Grasps: ", n_graps)
                print(self.succ_idx[0])

                # set the unused envs to IDLE
                all_envs = list(range(self.n_env))
                for i in all_envs[n_graps[0]:]:
                    self.envs[i].set_env_state(Environment.State.IDLE)

            # Shift all scenes
            for env in self.envs:
                env.shift_scene(offset=np.array([-40.0, 0.0, 0.0]))

        self.robot_observer.update(step_index, self.obj_id)
        self.obj_id, state, finished = self.environment_scheduler.step()
        # finished = True
        next_obj = self.prev_obj_id != self.obj_id
        
        # track the prev slip        
        if next_obj:
            self.slip[self.prev_obj_id] = self.robot_observer.get_slip()
            self.grasp[self.prev_obj_id] = self.grasp_single

            self.grasp_single = {}
            self.robot_observer.reset()

        if not finished:
            # Prepare for next Object
            if next_obj:
                print("Next Robot position: ", self.robot_positions[self.obj_id])
                print(f"obj {self.prev_obj_id} finished")

                self.robot_observer.set_object_2_observe(self.obj_id)
                print(state)

                # reset the prev envs for the next object
                for i in range(min(len(self.prev_state), self.n_env)):
                    self.reset_environment(i)

                # set the robot positions for the next object
                for i in range(min(len(state), self.n_env)):
                    env = self.envs[i]
                    
                    env.robot.set_default_state(self.robot_positions[self.obj_id] + self.work_positions[i])
                    env.articualtion.set_default_state(self.robot_positions[self.obj_id] + self.work_positions[i])
                    env.robot.post_reset()
                    env.articualtion.post_reset()
                    env.set_robot_state(Environment.RobotState.IDLE)

                # set the unused envs to IDLE
                all_envs = list(range(self.n_env))
                for i in all_envs[len(state):]:
                    self.envs[i].set_env_state(Environment.State.IDLE)

            print(f"Scene: {len(self.scene_paths)}/{self.scene_path_idx+1}")
            print(f"Object: {self.n_obj}/{self.obj_id+1}")
            env_states = [env.get_env_state().name for env in self.envs]
            print(env_states)
            print(state)

            ############################################################################################################

            # Apply Actions
            for i in range(min(len(state), self.n_env)):
                env_state = self.envs[i].get_env_state()
                robot_state = self.envs[i].get_robot_state()

                if env_state == Environment.State.TERMINATED or robot_state == Environment.RobotState.COLLISION:
                    continue

                #! Phase 0
                # Check whether all robots are already settled after they have moved to their target positions
                plan = self.cmd_plan[self.obj_id]
                if self._settle_time[i] < MAX_SETTLE_TIME:
                    self.envs[i].set_robot_state(Environment.RobotState.MOVING_TO_TARGET)
                    self.envs[i].set_env_state(Environment.State.RUNNING)

                    self.envs[i].set_grasp_pose_idx(state[i])

                    # apply action until settle
                    if self.cmd_idx_list[i] < len(plan[state[i]]):
                        self.envs[i].articualtion.apply_action(plan[state[i]][self.cmd_idx_list[i]])

                    # shift scene back and save current grasp pose
                    else:
                        self._settle_time[i] += T_INCREMENT_SETTLE
                        if self._settle_time[i] >= MAX_SETTLE_TIME:
                            if not self.envs[i].is_scene_permeable():
                                self.envs[i].shift_scene(offset=np.array([40.0, 0.0, 0.0]))
                                self.envs[i].scene._scene_permeability = True

                                self.r90 = R.from_euler('y', 90, degrees=True).as_matrix()
                                t, r = self.controller.fk(self.envs[i].articualtion.get_joint_positions()[:7])
                                t90 = np.dot(self.r90, t) + self.robot_positions[self.obj_id]  #  + self.work_positions[i]
                                rotation = np.dot(self.r90, r)
                                print(f"{i}.")
                                print(t90)
                                print(matrix_to_euler_angles(rotation, degrees=True, extrinsic=False))
                                grasps = self.envs[0].scene.get_grasps()[self.obj_id]
                                print(grasps.position[self.succ_idx[self.obj_id][i]].cpu())
                                print(matrix_to_euler_angles(grasps.rotation[self.succ_idx[self.obj_id][i]].cpu(), degrees=True, extrinsic=False))

                                T = np.eye(4)
                                T[:3, :3] = rotation
                                T[:3, 3] = t90

                                grasp_pose_idx = self.envs[i].get_grasp_pose_idx()
                                self.grasp_single[grasp_pose_idx] = T.tolist()
                                self.grasp_single = dict(sorted(self.grasp_single.items()))

                                self.sp = sphere.VisualSphere(
                                    prim_path="/curobo/robot_sphere_" + str(i),
                                    position=t90,
                                    orientation=rot_matrix_to_quat(rotation),
                                    radius=float(0.005),
                                    color=np.array([0, 0.8, 0.2]),
                                    scale=[0.1, 0.1, 0.1]
                                )

                self.cmd_idx_list[i] += 1

                #! Phase 1
                # Close the gripper for each robot
                if self._settle_time[i] >= MAX_SETTLE_TIME and self._t_close[i] < MAX_CLOSE_TIME:
                    self.envs[i].set_robot_state(Environment.RobotState.CLOSE)
                    
                    target_joint_positions = self.envs[i].robot.gripper.forward(action="close")
                    target_joint_positions.joint_velocities = [0, 0, 0, 0, 0, 0, 0, 0.1, 0.1]
                    self.envs[i].articualtion.apply_action(target_joint_positions)
                    self._t_close[i] += T_INCREMENT_CLOSE

                elif self._t_close[i] >= MAX_CLOSE_TIME and len(self._action_sequence[i]) == 0:
                    self._action_sequence[i] = self.controller.move_up_trajectory(self.envs[i].articualtion, None, i)

                    if len(self._action_sequence[i]) == 0:
                        grasp_pose_idx = self.envs[i].get_grasp_pose_idx()
                        self.robot_observer.remove_slip(grasp_pose_idx)
                        self.envs[i].set_robot_state(Environment.RobotState.FINISHED)
                        print(f"Env {i}")

                        continue

                #! Phase 2
                # Execute move_up action sequence
                if len(self._action_sequence[i]) != 0 and len(self._linear_action_sequence[i]) == 0:
                    if self._action_sequence_index[i] < len(self._action_sequence[i]):
                        self.envs[i].set_robot_state(Environment.RobotState.MOVE_UP)

                        action = self._action_sequence[i][self._action_sequence_index[i]]
                        action.joint_velocities[-1] = -0.1
                        action.joint_velocities[-2] = -0.1
                        self.envs[i].articualtion.apply_action(action)
                        self._action_sequence_index[i] += 1
                    else:
                        self._linear_action_sequence[i] = self.controller.linear_oscillation_trajectory(
                            self.envs[i].articualtion,
                            self._action_sequence[i][-1].joint_positions,
                            None,
                            i
                        )
                        if len(self._linear_action_sequence[i]) == 0:
                            self.envs[i].set_robot_state(Environment.RobotState.FINISHED)
                            print(f"Env {i}")
                            
                            continue

                        else:
                            if i == 0:
                                pos = self.collision_table.get_world_pose()[0]
                                new_pos = pos+np.array([-40.0, 0.0, 0.0])
                                self.collision_table.set_world_pose(new_pos)
                                self.collision_table.set_default_state(new_pos)

                            obj_idx = list(range(self.n_obj))
                            obj_idx.remove(self.obj_id)
                            self.envs[i].shift_scene(offset=np.array([-40.0, 0.0, 0.0]), obj_idx=obj_idx)

                #! Phase 3
                # Execute linear oscillation action sequence
                if len(self._linear_action_sequence[i]) != 0 and self._linear_action_sequence_index[i] < len(self._linear_action_sequence[i]):
                    self.envs[i].set_robot_state(Environment.RobotState.LINEAR)

                    action = self._linear_action_sequence[i][self._linear_action_sequence_index[i]]
                    action.joint_velocities[-1] = -0.1
                    action.joint_velocities[-2] = -0.1
                    self.envs[i].articualtion.apply_action(action)
                    self._linear_action_sequence_index[i] += 1
                
                #! Phase 4
                # Execute pendulum oscillation action sequence
                elif self._linear_action_sequence_index[i] >= len(self._linear_action_sequence[i]) and \
                     len(self._linear_action_sequence[i]) != 0 and \
                     self._period_pendulum_number[i] < MAX_PERIOD:

                    self.envs[i].set_robot_state(Environment.RobotState.PENDULUM)
                    
                    target_joint_positions, period_complete, self._reached_max_flags[i], self._reached_min_flags[i] = self.controller.pendulum_forward(
                                                                self.envs[i].robot.get_articulation_controller(),
                                                                self.envs[i].articualtion.get_joint_positions(),
                                                                self.envs[i].articualtion.get_joint_velocities(),
                                                                self._reached_max_flags[i], 
                                                                self._reached_min_flags[i]
                                                            )
                    self._period_pendulum_number[i] += period_complete

                    target_joint_positions.joint_velocities[-1] = -0.1
                    target_joint_positions.joint_velocities[-2] = -0.1
                    self.envs[i].articualtion.apply_action(target_joint_positions)
                
                #! Finish
                elif self._period_pendulum_number[i] >= MAX_PERIOD:
                    self.envs[i].set_robot_state(Environment.RobotState.FINISHED)

            ############################################################################################################

            # Clean up
            for i in range(min(len(state), self.n_env)):
                env = self.envs[i]
                robot_state = env.get_robot_state()
                env_state = env.get_env_state()

                # Vor dem letzten Grasp: Alle Environments zurücksetzen
                # oder
                # Nach dem letzten Grasp: Nur das letzte Environment zurücksetzen
                if robot_state in [Environment.RobotState.FINISHED, Environment.RobotState.COLLISION] and env_state != Environment.State.TERMINATED:
                    pose_idx = env.get_grasp_pose_idx()
                    last_grasp_reached = self.environment_scheduler.last_grasp_reached()
                    if last_grasp_reached and pose_idx in state:
                        self.terminate_environment(i)
                        print("Terminate Env: ", i)
                        env.set_env_state(Environment.State.TERMINATED)

                    else:
                        self.reset_environment(i)
                        print("Reset Env: ", i)
                        print(self.environment_scheduler.last_grasp_reached())
                        print(self.environment_scheduler.get_last_grasp_index())

                        env.set_env_state(Environment.State.FINISHED)
                        env.set_robot_state(Environment.RobotState.FINISHED)

            # track the obj_id for the next iteration
            self.prev_obj_id = self.obj_id
            self.prev_state = state
            
        else:
            print(f"Scene {self.scene_path_idx} Finished!")
            self.slip[self.obj_id] = self.robot_observer.get_slip()
            self.grasp[self.obj_id] = self.grasp_single
            
            self.grasp_single = {}
            self.robot_observer.reset()

            self.save_dataset()

            self.scene_path_idx += 1
            if self.scene_path_idx < len(self.scene_paths):
                self.clean_world()
                self.setup_world()
                self.config_robot()

                # Calculate an individual robot position for each object
                self.robot_positions = self.calculate_robot_positions(self.envs[0].scene, INIT_ROBOTER_POS, self.table_height)
                self.set_roboter_position(self.robot_positions[0])

                print("------Robot Positions-------")
                for rp in self.robot_positions:
                    print(rp)

                self.robot_orientations = [env.robot.get_local_pose()[1] for env in self.envs]
                self.controller = RobotController(
                                            name="collision_free_trajectory_generator",
                                            articulation=self.envs[0].articualtion,
                                            robot_positions=self.robot_positions,
                                            robot_orientations=self.robot_orientations,
                                            offset_position=np.array(self.base_work_position),
                                            batch=self.batch,
                                            warmup=self.warmup
                                )
                
                self.robot_observer = RobotObserver(envs=self.envs,
                                            work_positions=self.work_positions,
                                            robot_positions=self.robot_positions,
                                            controller=self.controller,
                                            leftfinger=self.joint_forces["panda_leftfinger"],
                                            rightfinger=self.joint_forces["panda_rightfinger"]
                                    )
                
                # TODO:
                self.obj_id = 0
                self.environment_scheduler = EnvironmentScheduler(envs=self.envs, start_id=self.obj_id)

                self._next_scene = True
                self.load_trajectorys = False

                print(f"Scene: {len(self.scene_paths)}/{self.scene_path_idx+1}")

            else:
                print("All Scenes Finished!")
                self._next_scene = False
                self._finished = True

        # self.controller.visulize_collsion_spheres(step_index, self.envs[0].articualtion)

        return
    
    def next_scene(self) -> bool:
        """
        """
        return self._next_scene
    
    def finished(self) -> bool:
        """
        """
        return self._finished

    def save_dataset(self) -> None:
        slip = {"grasp": self.grasp,
                "slip": self.slip}
        
        filename = os.path.join(self.scene_roots[self.scene_path_idx], "dataset" + ".json")
        with open(filename, 'w') as f:
            json.dump(slip, f)


'''
import omni.replicator.core as rep


camera_positions = [(0.85369,7.41818,1.48661)]

# Create the replicator camera
camera = rep.create.camera(position=(0.85369,7.41818,1.48661), look_at=(0.1001,6,0.82435))

# Set the renderer to Path Traced
# rep.settings.set_render_pathtraced(samples_per_pixel=256)

# Create the render product
render_product  = rep.create.render_product(camera, (1920*2, 1080*2))

# Initialize and attach writer
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir="_subframes_pt_example", rgb=True)
writer.attach([render_product])


# Render 3 frames, with 50 subframes
with rep.trigger.on_frame(num_frames=3, rt_subframes=150):
    with camera:
        f = 0
        #rep.modify.pose(position=rep.distribution.sequence(camera_positions))
'''
