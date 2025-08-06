import h5py
from matrix import T_Matrix
import numpy as np

from scene import Scene


from curobo.util_file import (
    join_path,
    load_yaml,
)


class SceneManager:
    def __init__(self, scene_config: str) -> None:
        self.scene_config = scene_config
        
        self._grasps = []
        self._object_paths = []
        
        self._prepare_grasps()
        self._prepare_objects()
        
    def _prepare_grasps(self):
        with h5py.File(self.grasps_directory, "r") as f:
            grasps = np.squeeze(f['dataset'][()])
            for _, (g1, g2) in enumerate(grasps):
                self._grasps.append((T_Matrix(g1), T_Matrix(g2)))
            print(grasps.shape)
        
    def _prepare_objects(self):
        self._object_paths.append(self.objects_directory)
        
    def get_grasp(self, index: int=0) -> T_Matrix:
        return self._grasps[index][0]
    