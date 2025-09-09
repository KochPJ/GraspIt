from omni.isaac.motion_generation import (
    LulaTaskSpaceTrajectoryGenerator,
    LulaKinematicsSolver,
    ArticulationTrajectory
)
from omni.isaac.core.controllers.articulation_controller import ArticulationController

from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.geom.sdf.world import CollisionCheckerType

from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    MotionGenResult
)
from scipy.spatial.transform import Rotation as R
from omni.isaac.core.utils.rotations import euler_angles_to_quat, rot_matrix_to_quat, quat_to_euler_angles, matrix_to_euler_angles
import lula
from omni.isaac.core.utils.extensions import get_extension_path_from_name
import os

from curobo.util.logger import log_error
from curobo.util.usd_helper import UsdHelper
from curobo.types.state import JointState
from curobo.types.math import Pose
from curobo.rollout.cost.pose_cost import PoseCostMetric

from omni.isaac.core.utils.rotations import quat_to_rot_matrix
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction, JointsState
from utils import get_robot_cfg, get_current_stage

from typing import List, Optional, Union
import torch
import numpy as np
from omni.isaac.core.objects import sphere
import curobo
import json

print(curobo.__version__)


class RobotController(object):
    def __init__(self, name: str,
                 articulation: Articulation,
                 robot_positions: Optional[List[np.ndarray]] = None, 
                 robot_orientations: Optional[List[np.ndarray]] = None,
                 offset_position: Optional[np.ndarray] = None, 
                 offset_orientation: Optional[np.ndarray] = None, 
                 batch: Optional[int] = None,
                 warmup: bool = None,
                 end_effector_initial_height: float = 0.3,
                 robot_config: dict = {} ) -> None:

        self.articulation = articulation
        self.batch = batch
        self.spheres = None
        self.cmd_plan = []

        self.usd_help = UsdHelper()
        self.usd_help.load_stage(get_current_stage())
        self.tensor_args = TensorDeviceType()

        robot_cfg = get_robot_cfg(robot_cfg_path=robot_config.get("robot_cfg_path"))
        j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
        default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

        world_cfg = WorldConfig()
        # self.usd_help.add_world_to_stage(world_cfg, base_frame=base_frame)

        # TODO: put in config file
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            [world_cfg],
            self.tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            num_trajopt_seeds=12,
            num_graph_seeds=32,
            interpolation_dt=0.01,
            collision_cache={"obb": 1, "mesh": 15},
            optimize_dt=True,
            trajopt_dt=None,
            trajopt_tsteps=32,
            trim_steps=None,
            ee_link_name='right_gripper',
            num_batch_trajopt_seeds=16,
            num_batch_ik_seeds=32,
            collision_activation_distance=0.013,
            maximum_trajectory_dt=0.05,
            interpolation_steps=500,
            # maximum_trajectory_dt=2.0,
            # self_collision_check=False
        )

        positions = []
        orientations = []

        # base_robot_position
        if not robot_positions:
            raise ValueError("robot_positions is empty.\n")
        for position in robot_positions:
            if position.shape == (3,):
                pose_tensor = self.tensor_args.to_device(torch.from_numpy(position))
                positions.append(pose_tensor)
            else:
                raise ValueError(
                    "Passed array is not of the right shape.\n"
                    "Expected shape (3,) but got array with shape {}.".format(self.base_robot_position.shape)
                )
        self.base_robot_position = torch.stack(positions)

        # offset_position
        if offset_position is not None:
            if offset_position.shape == (3,):
                self.offset_position = self.tensor_args.to_device(torch.from_numpy(offset_position))
            else:
                raise ValueError(
                    "Passed array is not of the right shape.\n"
                    "Expected shape (3,) but got array with shape {}.".format(offset_position.shape)
                )
        else:
            self.offset_position = self.tensor_args.to_device(torch.from_numpy(np.array([0, 0, 0])))

        # base_robot_orientation
        if not robot_orientations:
            raise ValueError("robot_orientations is empty.\n")
        for orientation in robot_orientations:
            if orientation.shape == (4,):
                rot_matrix = quat_to_rot_matrix(orientation)
                orientations.append(self.tensor_args.to_device(torch.from_numpy(rot_matrix)))
            else:
                raise ValueError(
                    "Passed array is not of the right shape.\n"
                    "Expected shape (4,) but got array with shape {}.".format(orientation.shape)
                )
        self.base_robot_orientation = torch.stack(orientations)

        # offset_orientation
        if offset_orientation is not None:
            if offset_orientation.shape == (4,):
                rot_matrix = quat_to_rot_matrix(offset_orientation)
                self.offset_orientation = self.tensor_args.to_device(torch.from_numpy(rot_matrix))
            else:
                raise ValueError(
                    "Passed array is not of the right shape.\n"
                    "Expected shape (4,) but got array with shape {}.".format(offset_orientation.shape)
                )
        else:
            rot_matrix = quat_to_rot_matrix(np.array([1, 0, 0, 0]))
            self.offset_orientation = self.tensor_args.to_device(torch.from_numpy(rot_matrix))

        self.motion_gen = MotionGen(motion_gen_config)
        
        # TODO: put in config file
        self.plan_config = MotionGenPlanConfig(
            enable_graph=True,
            enable_graph_attempt=30,
            max_attempts=60,
            enable_finetune_trajopt=True,
            parallel_finetune=True,
            # time_dilation_factor=0.5,
            pose_cost_metric=PoseCostMetric(reach_full_pose=True),
            num_trajopt_seeds=32,
            # num_ik_seeds=50,
            # enable_opt=False
            timeout=10,
            num_graph_seeds=16,
        )

        if warmup:
            self.warmup(batch=batch)

        self._end_effector_initial_height = end_effector_initial_height

        # taskspace_trajectory_generator controller
        mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
        kinematics_config_dir = os.path.join(
            mg_extension_path, "motion_policy_configs"
        )
        self._taskspace_trajectory_generator = LulaTaskSpaceTrajectoryGenerator(
            robot_description_path=robot_config.get("robot_description_path"),
            urdf_path=robot_config.get("urdf_path"),
        )
        self._taskspace_trajectory_generator.get_path_conversion_config().max_iterations = 1000
        self._taskspace_trajectory_generator.get_path_conversion_config().max_position_deviation = 0.01

        # Kinematic Solver
        print(kinematics_config_dir)
        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path=robot_config.get("robot_description_path"),
            urdf_path=robot_config.get("urdf_path"),
        )

        self.r90 = R.from_euler('y', 90, degrees=True).as_matrix()

        # TODO: put in config file
        # Pendulum Config
        self.speed = np.pi/2
        self.max_speed = 2 * np.pi / 2
        self._max_joint_position = 3.7525
        self._min_joint_position = self._max_joint_position - np.pi
        self._period_complete = False

    @property
    def get_js_names(self) -> List[str]:
        return self.articulation.dof_names

    def warmup(self, batch: Optional[int] = None) -> None:
        print("warming up...")
        self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False, parallel_finetune=True, batch=batch)
        print("Curobo is Ready")
        
        return
        
    def get_obstacles(self, only_paths: Optional[List[str]] = None, ignore_paths: Optional[List[str]] = None) -> WorldConfig:
        obstacles = self.usd_help.get_obstacles_from_stage(only_paths=only_paths, 
                                                           ignore_paths=ignore_paths).get_collision_check_world()
        return obstacles
        
    def update_world(self, obstacles) -> None:
        self.motion_gen.update_world(obstacles)

    def reset_seed(self):
        self.motion_gen.reset_seed()
    
    def plan(self, ik_goals: Union[Pose, List[Pose]], joints_state: JointsState=None) -> tuple[List[List[List[ArticulationAction]]], List[np.ndarray]]:
        """
        """
        
        if isinstance(ik_goals, Pose):
            ik_goal =  [ik_goal]

        if joints_state is None:
            joints_state = self.articulation.get_joints_state()

        if np.any(np.isnan(joints_state.positions)):
            log_error("isaac sim has returned NAN joint position values.")

        cu_js = JointState(
            position=self.tensor_args.to_device(joints_state.positions),
            velocity=self.tensor_args.to_device(joints_state.velocities),
            acceleration=self.tensor_args.to_device(joints_state.velocities),
            jerk=self.tensor_args.to_device(joints_state.velocities),
            joint_names=self.get_js_names,
        )
        cu_js.velocity *= 0.0
        cu_js.acceleration *= 0.0
        cu_js.jerk *= 0.0
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)
        cu_js = cu_js.unsqueeze(0).repeat_seeds(self.batch)
        
        # ik_goals = [ik_goals[2]]

        all_cmd_plans = []
        all_idx_lists = []

        for ik_goal_idx, ik_goal in enumerate(ik_goals):
            cmd = []
            idx = []
            n_poses = len(ik_goal.position)
            print(f"{ik_goal_idx}: n_poses: ", n_poses)

            # Update robot position for each object
            self.motion_gen.robot_cfg.kinematics.kinematics_config.fixed_transforms[0, :3, 3] = self.base_robot_position[ik_goal_idx] + self.offset_position
            self.motion_gen.robot_cfg.kinematics.kinematics_config.fixed_transforms[0, :3, :3] = self.base_robot_orientation[ik_goal_idx]
            
            for i in range(int(np.ceil(n_poses / self.batch))):
                start_idx = i * self.batch
                end_idx = min((i + 1) * self.batch, n_poses)
                position = ik_goal.position[start_idx:end_idx]
                quaternion = ik_goal.quaternion[start_idx:end_idx]
                print(start_idx, ":", end_idx," = ",start_idx-end_idx)
                print("len(position):", len(position))
                
                # check if padding is needed
                if len(position) < self.batch:
                    additional_poses = self.batch - len(position)
                    placeholder_pose = self.base_robot_position[ik_goal_idx]+self.tensor_args.to_device(torch.tensor([0.45, 0, 0.3]))
                    print(placeholder_pose)
                    placeholder_quat = self.tensor_args.to_device(euler_angles_to_quat(np.array([0, np.pi, 0])))
                    position = torch.concatenate((position, torch.tile(placeholder_pose, (additional_poses, 1))))
                    quaternion = torch.concatenate((quaternion, torch.tile(placeholder_quat, (additional_poses, 1))))

                goal_pose = Pose(position=position, quaternion=quaternion, normalize_rotation=False)
                print(goal_pose.batch)
                start_state = cu_js.trim_trajectory(start_idx=0, end_idx=self.batch)
                
                while True:
                    try:
                        result = self.motion_gen.plan_batch(start_state, goal_pose, self.plan_config)
                        break
                    except Exception as e:
                        print(f"Fehler aufgetreten: {e}. Versuche es erneut...")
                
                # result = self.motion_gen.plan_batch(start_state, goal_pose, self.plan_config)
                
                # remove padded results
                if len(position) > end_idx - start_idx:
                    result.success = result.success[:end_idx - start_idx]
                
                idx.extend(result.success.cpu().tolist())

                if torch.count_nonzero(result.success) > 0:
                    cmd_batch = self._get_full_js(result)[:end_idx - start_idx]
                    actions_batch = [self._to_articulation_actions(c) for c in cmd_batch]
                    print(f"{len(actions_batch)} grasps found.")

                    cmd.extend(actions_batch)

            self.motion_gen.reset_seed()
            all_cmd_plans.append(cmd)
            all_idx_lists.append(np.where(np.array(idx))[0])

        print('Finish')
        return all_cmd_plans, all_idx_lists
    
    def save_trajectorys(self, cmd_plans: List[List[List[ArticulationAction]]], idx_lists: List[np.ndarray], filename: str ='output.json'):
        """
        """
        serialized_data = {
            'cmd_plans': [[[action.get_dict() for action in batch] for batch in cmd_plan] for cmd_plan in cmd_plans],
            'idx_lists': [idx.tolist() for idx in idx_lists],
            'robot_positions': [pos.tolist() for pos in self.base_robot_position],
            'robot_orientation': [ori.tolist() for ori in self.base_robot_orientation]
        }
        
        with open(filename, 'w') as f:
            json.dump(serialized_data, f, indent=4)

    def dict_to_articulation_action(self, data: dict) -> ArticulationAction:
        """
        """
        return ArticulationAction(
            joint_positions=np.array(data.get("joint_positions")) if data.get("joint_positions") is not None else None,
            joint_velocities=np.array(data.get("joint_velocities")) if data.get("joint_velocities") is not None else None,
            joint_efforts=np.array(data.get("joint_efforts")) if data.get("joint_efforts") is not None else None
        )

    def load_trajectorys(self, filename: str ='output.json') -> tuple[List[List[List[ArticulationAction]]], List[np.ndarray]]:
        """
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        cmd_plans = [
            [
                [self.dict_to_articulation_action(action_dict) for action_dict in batch] for batch in cmd_plan
            ]
            for cmd_plan in data['cmd_plans']
        ]
        
        idx_lists = [np.array(idx_list) for idx_list in data['idx_lists']]
        
        return cmd_plans, idx_lists
    
    def _get_full_js(self, result: MotionGenResult) -> List[JointState]:
        """
        """
        trajs = result.get_paths()
        cmd = []
        for _, s in enumerate(range(len(result.success))):
            if result.success[s]:
                cmd.append(self.motion_gen.get_full_js(trajs[s]))
                
                # get only joint names that are in both:
                common_js_names = []
                for x in self.get_js_names:
                    if x in cmd[-1].joint_names:
                        common_js_names.append(x)

                cmd[-1] = cmd[-1].get_ordered_joint_state(common_js_names)
        
        return cmd
    
    def _to_articulation_actions(self, cmd_plan: List[JointState]) -> List[ArticulationAction]:
        """
        """
        return [ArticulationAction(cmd_plan[i].position.cpu().numpy(),
                                    cmd_plan[i].velocity.cpu().numpy()) for i in range(len(cmd_plan))]
    
    def _joint_state_to_device(self, joints_state: JointsState) -> JointsState:
        """
        """
        cu_js = JointState(
            position=self.tensor_args.to_device(joints_state.positions),
            velocity=self.tensor_args.to_device(joints_state.velocities),
            acceleration=self.tensor_args.to_device(joints_state.velocities),
            jerk=self.tensor_args.to_device(joints_state.velocities),
            joint_names=self.get_js_names,
        )
        cu_js.velocity *= 0.0
        cu_js.acceleration *= 0.0
        cu_js.jerk *= 0.0
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)
        return cu_js
    
    def move_up_trajectory(self, 
                           articulation: Articulation, 
                           end_effector_initial_height: float = None,
                           i: int = 0) -> List[ArticulationAction]:
        """
        """
        if end_effector_initial_height is None:
            end_effector_initial_height = self._end_effector_initial_height

        t0, r = self.fk(articulation.get_joint_positions()[:7])
        r0 = rot_matrix_to_quat(r)

        w, x, y, z = r0
        r0 = lula.Rotation3(w, x, y, z)

        composite_path_spec = lula.create_composite_path_spec(articulation.get_joint_positions()[:7])
        task_space_spec = lula.create_task_space_path_spec(lula.Pose3(r0, t0))
        
        for _ in range(10):
            t0[0] -= 0.02
            task_space_spec.add_translation(t0)
        
        for _ in range(10):
            t0[2] += 0.02
            task_space_spec.add_translation(t0)
        
        '''
        for _ in range(10):
            t0[1] += 0.02
            task_space_spec.add_translation(t0)
        
        vz = 1
        for _ in range(2):
            for _ in range(10):
                t0[1] = t0[1] - vz * 0.04
                task_space_spec.add_translation(t0)
            vz *= -1
        '''
        
        transition_mode = lula.CompositePathSpec.TransitionMode.FREE
        composite_path_spec.add_task_space_path_spec(task_space_spec, transition_mode)

        trajectory = None
        try:
            trajectory = self._taskspace_trajectory_generator.compute_task_space_trajectory_from_path_spec(
                composite_path_spec, "right_gripper"
            )
        except Exception as e:
            print(f"Fehler aufgetreten: {e}. Versuche es erneut...")
        
        action_sequence = None
        if trajectory is None:
            print("No trajectory could be computed in Move_Up")
            action_sequence = []
        else:
            physics_dt = 1/60
            articulation_trajectory = ArticulationTrajectory(articulation, trajectory, physics_dt)
            action_sequence = articulation_trajectory.get_action_sequence()
        
        return action_sequence
    
    def linear_oscillation_trajectory(self,
                        articulation: Articulation, 
                        joint_positions: np.ndarray = None,
                        end_effector_initial_height: float = None, 
                        i: int = 0) -> List[ArticulationAction]:
        """
        """
        if end_effector_initial_height is None:
            end_effector_initial_height = self._end_effector_initial_height

        if joint_positions is None:
            joint_positions = articulation.get_joint_positions()[:7]

        t0, r = self.fk(joint_positions)
        r0 = rot_matrix_to_quat(r)

        w, x, y, z = r0
        r0 = lula.Rotation3(w, x, y, z)

        composite_path_spec = lula.create_composite_path_spec(joint_positions)
        task_space_spec = lula.create_task_space_path_spec(lula.Pose3(r0, t0))

        for _ in range(10):
            t0[1] += 0.02
            task_space_spec.add_translation(t0)
        
        vz = 1
        for _ in range(2):
            for _ in range(10):
                t0[1] = t0[1] - vz * 0.04
                task_space_spec.add_translation(t0)
            vz *= -1
        
        transition_mode = lula.CompositePathSpec.TransitionMode.FREE
        composite_path_spec.add_task_space_path_spec(task_space_spec, transition_mode)

        trajectory = self._taskspace_trajectory_generator.compute_task_space_trajectory_from_path_spec(
            composite_path_spec, "right_gripper"
        )
        
        action_sequence = None
        if trajectory is None:
            print("No trajectory could be computed in Linear_osc")
            action_sequence = []
        else:
            physics_dt = 1/60
            articulation_trajectory = ArticulationTrajectory(articulation, trajectory, physics_dt)
            action_sequence = articulation_trajectory.get_action_sequence()
    
        return action_sequence

    def pendulum_forward(
        self,
        franka_art_controller: ArticulationController,
        current_joint_positions: Optional[np.ndarray],
        current_joint_velocities: Optional[np.ndarray],
        reached_max: bool,
        reached_min: bool,
    ) -> tuple[ArticulationAction, bool, bool, bool]:
        """
        """
        franka_art_controller.switch_dof_control_mode(
            dof_index=5, mode="velocity"
        )
        target_joint_velocities = [None] * current_joint_velocities.shape[0]

        # Überprüfen, ob die max oder min Position erreicht wurde
        if current_joint_positions[5] > self._max_joint_position:
            reached_max = True
            spin_left = False
        elif current_joint_positions[5] < self._min_joint_position:
            reached_min = True
            spin_left = True
        else:
            # Beibehalten der aktuellen Geschwindigkeit
            spin_left = current_joint_velocities[5] > 0

        # Prüfen, ob beide Endpunkte erreicht wurden
        period_complete = reached_max and reached_min

        # Geschwindigkeit basierend auf der Drehrichtung setzen
        if spin_left:
            target_joint_velocities[5] = self.speed
        else:
            target_joint_velocities[5] = -self.speed

        # Wenn eine Periode abgeschlossen ist, zurücksetzen
        if period_complete:
            reached_max = False
            reached_min = False

        target_joints = ArticulationAction(joint_velocities=target_joint_velocities)

        return target_joints, period_complete, reached_max, reached_min
    
    def spin_ee_forward(
        self,
        franka_art_controller: ArticulationController,
        current_joint_positions: Optional[np.ndarray],
        current_joint_velocities: Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """
        """
        franka_art_controller.switch_dof_control_mode(
                dof_index=6, mode="velocity"
        )
        target_joint_velocities = [None] * current_joint_velocities.shape[0]

        target_joint_velocities[6] = self.speed

        target_joints = ArticulationAction(joint_velocities=target_joint_velocities)

        return target_joints

    def fk(self, joint_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        """
        position, rotation = self._kinematics_solver.compute_forward_kinematics(
            "right_gripper", joint_positions[:7]
        )
        return position, rotation

    def visulize_collsion_spheres(self, step_index, articulation: Articulation) -> None:
        """
        """
        if step_index % 2 == 0:
            joints_state = articulation.get_joints_state()
            js_names = articulation.dof_names
            if np.any(np.isnan(joints_state.positions)):
                log_error("isaac sim has returned NAN joint position values.")
            cu_js = JointState(
                position=self.tensor_args.to_device(joints_state.positions),
                velocity=self.tensor_args.to_device(joints_state.velocities),
                acceleration=self.tensor_args.to_device(joints_state.velocities),
                jerk=self.tensor_args.to_device(joints_state.velocities),
                joint_names=js_names,
            )
            cu_js.velocity *= 0.0
            cu_js.acceleration *= 0.0
            cu_js.jerk *= 0.0
            cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)
            cu_js = cu_js.unsqueeze(0).repeat_seeds(self.batch)
            sph_list = self.motion_gen.kinematics.get_robot_as_spheres(cu_js.position)

            if self.spheres is None:
                self.spheres = []
                # create spheres:

                for si, s in enumerate(sph_list[0]):
                    sp = sphere.VisualSphere(
                        prim_path="/curobo/robot_sphere_" + str(si),
                        position=np.ravel(s.position),
                        radius=float(s.radius),
                        color=np.array([0, 0.8, 0.2]),
                    )
                    self.spheres.append(sp)
            else:
                for si, s in enumerate(sph_list[0]):
                    if not np.isnan(s.position[0]):
                        self.spheres[si].set_world_pose(position=np.ravel(s.position))
                        self.spheres[si].set_radius(float(s.radius))
    
    def reset(self) -> None:
        """
        """
        self.motion_gen.reset(reset_seed=True)



"""
pytroch 11.8: pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

pytorch 12.1: pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pytorch 12.1: pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pytorch 12.1: pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pytorch 12.1: pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

pytorch 12.4: pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
"""
