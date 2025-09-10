import trimesh
import numpy as np
from typing import Optional, List, Union
from scipy.spatial.transform import Rotation as R
from utlis import usd2trimesh, usd2obj, quat_to_rot_matrix
import os


class Object(object):
    def __init__(self, 
                 name: str = "object", 
                 file_path: str = None,
                 orientation: Optional[Union[List, np.ndarray]] = None,
                 position: Optional[Union[List, np.ndarray]] = None,
                 scale: Optional[Union[List, np.ndarray]] = None
                 ):
        self.name = name
        
        #if not file_path.startswith('/'):
        #    file_path = '/' + file_path
        #    print(file_path)
        

        try:
            self.mesh = trimesh.load(file_path)
            print(file_path)
        except:
            root, ext = os.path.splitext(file_path)
            if ext in [".usd"]:
                '''
                output_obj = root + ".obj"
                print(file_path)
                print(output_obj)
                usd2obj(file_path, output_obj)
                self.mesh = trimesh.load(output_obj)
                '''
                self.mesh = usd2trimesh(file_path)
                
            else:
                
                raise NameError(f"No usd file, {file_path}.")

        if scale is not None:
            self.rescale(scale)

        if position is not None and orientation is not None:
            self.transform(position, orientation)

    def rescale(self, scale: Union[List, np.ndarray]):
        self.mesh.apply_scale(scale)

    def transform(self, 
                  position: Optional[Union[List, np.ndarray]] = None,
                  orientation: Optional[Union[List, np.ndarray]] = None,
                  return_tran=False):
        
        T = np.eye(4)
        T[:3, :3] = quat_to_rot_matrix(orientation)
        T[:3, 3] = position
        self.mesh.apply_transform(T)

        if return_tran:
            return T
        