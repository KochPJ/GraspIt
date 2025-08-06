from typing import List
import numpy as np
from scipy.spatial.transform import Rotation as R

from omni.isaac.core.prims import RigidPrimView
import omni.isaac.core.utils.prims as prims_utils
from omni.isaac.debug_draw import _debug_draw
from omni.isaac.core.objects import sphere, cuboid
from omni.isaac.core.utils.rotations import (
    rot_matrix_to_quat, 
    quat_to_rot_matrix, 
    matrix_to_euler_angles,
    euler_angles_to_quat,
)

from environment import Environment
from controller2 import RobotController
from omni.isaac.core.prims import XFormPrim

from omni.usd.commands import MovePrimsCommand, MovePrimCommand

from omni.isaac.core.utils.stage import add_reference_to_stage, open_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path


class RobotObserver(object):
    """
    """
    def __init__(self,
                 envs: List[Environment],
                 work_positions: List[np.ndarray],
                 robot_positions: List[np.ndarray],
                 controller: RobotController,
                 leftfinger: RigidPrimView,
                 rightfinger: RigidPrimView):
        self.envs = envs
        self.work_positions = work_positions
        self.robot_positions = robot_positions
        self.controller = controller
        self.contact_left = leftfinger
        self.contact_right = rightfinger

        self.initial_pose = envs[0].scene.get_init_pose()
        print("initial_pose Objects:\n")
        for p, o in self.initial_pose:
            print("Pos: ", p)
            print("Ori: ", o)
            print('-----------------------------------------')

        self.draw = _debug_draw.acquire_debug_draw_interface()

        self.r90 = R.from_euler('y', 90, degrees=True).as_matrix()

        self.ii = 0
        self.sp = None

        self.tcps : List[XFormPrim] = []
        self.refs : List[XFormPrim] = []
        self.refs_obj : List[XFormPrim] = []

        self.tcps_obj_distance = [None]*len(envs)

        num_envs = len(envs)
        XFormPrim(f"/World/tcp")
        for i in range(num_envs):
            # add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", f"/World/tcp/tcp_{i}")
            self.tcps.append(XFormPrim(f"/World/tcp/tcp_{i}", scale=[.04,.04,.04]))
            self.refs.append(XFormPrim(f"/World/tcp/tcp_{i}/referenz"))
            self.refs_obj.append(XFormPrim(envs[i].scene.objects[0].prim_path + "/referenz"))

        self.num_tests = 4
        self.max_slip_ori = np.full([num_envs, self.num_tests, 3], None)
        self.max_slip_pos = np.full([num_envs, self.num_tests, 3], None)

        self.slip = {}

    def set_object_2_observe(self, obj_id: int) -> None:
        """
        """
        for i, env in enumerate(self.envs):
            prims_utils.delete_prim(self.refs_obj[i].prim_path)
            self.refs_obj[i] = XFormPrim(env.scene.objects[obj_id].prim_path + "/referenz")

        return
    
    def calculate_slip(self, env_id: int, obj_id: int, tcp_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        """
        # Tranformationmatrix of current object
        pose_obj = np.eye(4)
        pose_obj[:3, :3] = quat_to_rot_matrix(self.refs_obj[env_id].get_world_pose()[1])
        pose_obj[:3, 3]  = self.refs_obj[env_id].get_world_pose()[0]

        # Tranformationmatrix of current robot tcp
        pose_tcp = np.eye(4)
        pose_tcp[:3, :3] = quat_to_rot_matrix(self.refs[env_id].get_world_pose()[1])
        pose_tcp[:3, 3]  = self.refs[env_id].get_world_pose()[0]

        slip = np.linalg.inv(pose_tcp) @ pose_obj
        oriantion_slip = matrix_to_euler_angles(slip[:3, :3], degrees=True, extrinsic=False)

        tcp_obj_distance_current = tcp_pos - self.envs[env_id].scene.objects[obj_id].get_world_pose()[0]
        position_slip = self.tcps_obj_distance[env_id] - tcp_obj_distance_current

        return position_slip, oriantion_slip

    def update(self, step_index: int, obj_id: int) -> None:
        for i in range(len(self.envs)):
            robot_state = self.envs[i].get_robot_state()
            env_state = self.envs[i].get_env_state()

            if env_state == Environment.State.IDLE:
                continue

            t, r = self.controller.fk(self.envs[i].articualtion.get_joint_positions()[:7])
            t90 = np.dot(self.r90, t) + self.robot_positions[obj_id] + self.work_positions[i]
            rotation = np.dot(self.r90, r)

            self.tcps[i].set_world_pose(position=t90, orientation=rot_matrix_to_quat(rotation))

            if robot_state in [Environment.RobotState.MOVE_UP, Environment.RobotState.LINEAR, Environment.RobotState.PENDULUM]:
                contacts_left = self.contact_left.get_net_contact_forces(clone=False)[i]
                contacts_right = self.contact_right.get_net_contact_forces(clone=False)[i]
                if np.all(contacts_left == 0) and np.all(contacts_right == 0):
                    grasp_pose_idx = self.envs[i].get_grasp_pose_idx()
                    if grasp_pose_idx in self.slip:
                        if robot_state == Environment.RobotState.MOVE_UP:
                            self.remove_slip(grasp_pose_idx)

                        elif robot_state == Environment.RobotState.LINEAR:
                            self.max_slip_pos[i][2:] = None
                            self.max_slip_ori[i][2:] = None
                            self._add_slip(i, grasp_pose_idx)

                        elif robot_state == Environment.RobotState.PENDULUM:
                            self.max_slip_pos[i][3:] = None
                            self.max_slip_ori[i][3:] = None
                            self._add_slip(i, grasp_pose_idx)

                    self.tcps_obj_distance[i] = None
                    self.max_slip_pos[i][:] = None
                    self.max_slip_ori[i][:] = None

                    print("Collision Robot: ", i)
                    print("Env finished: ", i)
                    self.envs[i].set_robot_state(Environment.RobotState.COLLISION)

                    continue

            if robot_state == Environment.RobotState.MOVING_TO_TARGET:
                self.refs[i].set_world_pose(orientation=euler_angles_to_quat(np.array([0, 0, 0]), True))
                self.refs_obj[i].set_world_pose(orientation=euler_angles_to_quat(np.array([0, 0, 0]), True))

            # Abstand zwischen TCP und Objektposition initialisieren
            elif robot_state == Environment.RobotState.CLOSE and self.tcps_obj_distance[i] is None:
                self.tcps_obj_distance[i] = t90 - self.envs[i].scene.objects[obj_id].get_world_pose()[0]

            if robot_state in [Environment.RobotState.CLOSE,
                               Environment.RobotState.MOVE_UP,
                               Environment.RobotState.LINEAR,
                               Environment.RobotState.PENDULUM]:
                position_slip, oriantion_slip = self.calculate_slip(env_id=i, obj_id=obj_id, tcp_pos=t90)

                # Determine the maximum slip
                for idx in range(3):
                    robot_state_idx = Environment.RobotState.get_index(robot_state)
                    # Update max slip position
                    max_pos = self.max_slip_pos[i][robot_state_idx-1][idx]
                    if max_pos is None or position_slip[idx] > max_pos:
                        self.max_slip_pos[i][robot_state_idx-1][idx] = position_slip[idx]
                    
                    # Update max slip orientation
                    max_ori = self.max_slip_ori[i][robot_state_idx-1][idx]
                    if max_ori is None or oriantion_slip[idx] > max_ori:
                        self.max_slip_ori[i][robot_state_idx-1][idx] = oriantion_slip[idx]

                grasp_pose_idx = self.envs[i].get_grasp_pose_idx()
                self._add_slip(i, grasp_pose_idx)

                # point_list_1 = [(t90)]
                # self.draw.draw_points(point_list_1, [(1, 0, 0, 1)] * 1, [10] * 1)
                '''
                self.sp = cuboid.VisualCuboid(
                        prim_path="/curobo/robot_sphere_" + str(self.ii),
                        position=t90,
                        orientation=rot_matrix_to_quat(np.dot(r_rot, R_90)),
                        #radius=float(0.005),
                        color=np.array([0, 0.8, 0.2]),
                        scale=[0.1, 0.1, 0.1]
                )
                '''
                # self.sp.set_world_pose(orientation=rot_matrix_to_quat(np.dot(r_rot, R_90)))
                # sp.set_radius(float(0.005)
                # self.ii += 1

            elif env_state == Environment.State.FINISHED:
                self.tcps_obj_distance[i] = None
                self.max_slip_pos[i][:] = None
                self.max_slip_ori[i][:] = None

                print("Env finished---: ", i)

    def _add_slip(self, env_id: int, grasp_pose_idx: int) -> None:
        """
        """
        slip = np.full((2, self.num_tests, 3), None)
        slip[0] = self.max_slip_pos[env_id]
        slip[1] = self.max_slip_ori[env_id]

        self.slip[grasp_pose_idx] = slip.tolist()
        self.slip = dict(sorted(self.slip.items()))

        return
    
    def remove_slip(self, grasp_pose_idx: int) -> None:
        """
        """
        if grasp_pose_idx in self.slip:
            self.slip.pop(grasp_pose_idx)

        return

    def reset(self):
        """
        """
        self.slip = {}

        for i in range(len(self.envs)):
            self.tcps_obj_distance[i] = None
            self.max_slip_pos[i][:] = None
            self.max_slip_ori[i][:] = None

    def get_slip(self) -> dict:
        """
        """
        return self.slip
