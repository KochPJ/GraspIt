from __future__ import annotations

from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.prims import delete_prim
from pxr import Usd

from typing import Optional, Union, Tuple, Sequence, List
import numpy as np
import torch

from scene import Scene

from omni.isaac.franka import Franka
from omni.isaac.core.articulations import Articulation

from enum import Enum


class Environment(XFormPrim):
    """
    """
      
    class State(Enum):
        IDLE = 'idle'
        FINISHED = 'finished'
        TERMINATED = 'terminated'
        RUNNING = 'running'

        @classmethod
        def get_index(cls, state):
            return list(cls).index(state)
        
    # TODO: in Robot Class
    class RobotState(Enum):
        MOVING_TO_TARGET = "moving_to_target"
        CLOSE = "close"
        MOVE_UP = "move_up"
        LINEAR = "linear"
        PENDULUM = "pendulum"
        IDLE = 'idle'
        FINISHED = 'finished'
        COLLISION = 'collision'

        @classmethod
        def get_index(cls, state):
            return list(cls).index(state)
            
    def __init__(self, 
                 prim_path: str,        
                 name: str,
                 scene: Scene,
                 robot: Franka,
                 robot_articualtion: Articulation,
                 position: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 orientation: Optional[Union[np.ndarray, torch.Tensor]] = None):
        
        super().__init__(prim_path=prim_path, name=name, position=position, orientation=orientation)
        
        self.scene = scene
        self.robot = robot
        self.articualtion = robot_articualtion
        
        self._grasp_pose_idx = None

        self._state = Environment.State.IDLE
        self._robot_state = Environment.RobotState.IDLE

        return
    
    def get_env_state(self) -> Environment.State:
        """
        """
        return self._state

    def set_env_state(self, state: Environment.State) -> None:
        """
        Sets the environment state.
        
        :param state: The state to be set. This must be a string.
        :raises TypeError: If the state is not of type string.
        """
        if not isinstance(state, Environment.State):
            raise TypeError(f"Invalid type: {type(state).__name__}. 'state' must be a Environment.State.")
        self._state = state
        
    def get_robot_state(self) -> Environment.RobotState:
        """
        """
        return self._robot_state
    
    def set_robot_state(self, state: Environment.RobotState):
        """
        Sets the robot state.
        
        :param state: The state to be set. This must be a string.
        :raises TypeError: If the state is not of type string.
        """
        if not isinstance(state, Environment.RobotState):
            raise TypeError(f"Invalid type: {type(state).__name__}. 'state' must be a Environment.RobotState.")
        self._robot_state = state

    def get_grasp_pose_idx(self) -> int:
        """
        """
        return self._grasp_pose_idx
    
    def set_grasp_pose_idx(self, idx: int):
        """
        """
        self._grasp_pose_idx = idx
    
    def get_world_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        """
        return self.get_world_pose()
    
    def get_local_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        """
        return self.get_local_pose()
    
    def set_world_pose(
        self, position: Optional[Sequence[float]] = None, orientation: Optional[Sequence[float]] = None
    ) -> None:
        """
        """
        self.set_world_pose(position=position, orientation=orientation)
        return
    
    def set_local_pose(
        self, translation: Optional[Sequence[float]] = None, orientation: Optional[Sequence[float]] = None
    ) -> None:
        """
        """
        self.set_local_pose(translation=translation, orientation=orientation)
        return
    
    def initialize(self, physics_sim_view=None) -> None:
        """
        """
        super().initialize(physics_sim_view)
        self.scene.initialize(physics_sim_view)
        self.robot.initialize(physics_sim_view)
        self.articualtion.initialize(physics_sim_view)
        return
    
    def activate_scene_permeability(self) -> None:
        """
        """
        self.scene.activate_scene_permeability()
        return

    def deactivate_scene_permeability(self) -> None:
        """
        """
        self.scene.deactivate_scene_permeability()
        return
    
    def shift_scene(self, offset: Optional[np.ndarray] = None, obj_idx: List = None):
        """
        """
        self.scene.shift(offset=offset, obj_idx=obj_idx)
    
    def is_scene_permeable(self) -> bool:
        """
        """
        return self.scene.is_permeable()
    
    def delete_scene(self):
        """
        """
        self.scene.delete()
    
    @property
    def prim(self) -> Usd.Prim:
        """
        """
        return self.prim
    
    def delete(self):
        """
        """
        self.scene.delete()
        delete_prim(self.robot.prim_path)
        delete_prim(self.prim_path)
        return


# TODO: envs
class EnvironmentScheduler(object):
    """
    """
    def __init__(self, envs: List[Environment], start_id: int = 0):
        self.envs = envs
        self.initial_pose = envs[0].scene.get_init_pose()
        self.states = list(range(0, len(envs))) 
        self.next_state = len(envs)+1

        '''self.states = list(range(145, 170)) 
        self.next_state = 150'''

        self._n_grasp = None
        self.obj_id = start_id
        self._current_obj_finished = True
        self._last_grasp_index = None

    def step(self):
        """
        """
        self._current_obj_finished = True

        for i in range(min(self._n_grasp[self.obj_id], len(self.envs))):
            env_state = self.envs[i].get_env_state()

            last_grasp_reached = self.last_grasp_reached()
            
            if env_state == Environment.State.FINISHED and not last_grasp_reached:
                print('hererer')
                self.states[i] = self.next_state
                self.next_state += 1
                if self.last_grasp_reached():
                    self._last_grasp_index = i
            elif env_state in [Environment.State.RUNNING, Environment.State.IDLE] or not last_grasp_reached:
                self._current_obj_finished = False

        if self._current_obj_finished:
            self._advance_obj_id()

        finished = self._finished()
        
        return self.obj_id, self.states, finished

    def _advance_obj_id(self):
        """
        """
        while True:
            self.obj_id += 1
            if self.obj_id >= len(self._n_grasp):
                self.obj_id = len(self._n_grasp) - 1
                break

            elif self._n_grasp[self.obj_id] != 0:
                self.reset()
                break

    def _finished(self) -> bool:
        """
        """
        return self._current_obj_finished and self.obj_id == len(self._n_grasp)-1
    
    def reset(self) -> list:
        """
        """
        self.states = list(range(0, min(self._n_grasp[self.obj_id], len(self.envs))))
        self.next_state = len(self.envs)+1
        self._current_obj_finished = False
        self._last_grasp_index = None

        '''self.states = list(range(145, 170)) 
        self.next_state = 150'''
        
        return self.states

    def set_num_grasp(self, n_graps: List[int]):
        """
        """
        self._n_grasp = n_graps

    def current_obj_finished(self) -> bool:
        """
        """
        return self._current_obj_finished
    
    def get_last_grasp_index(self) -> int:
        """
        """
        return self._last_grasp_index

    def last_grasp_reached(self) -> bool:
        """
        """
        return self.next_state >= self._n_grasp[self.obj_id]
