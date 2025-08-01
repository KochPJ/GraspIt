import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R


class Gripper(object):
    def __init__(self, custom: bool = False, gripper_path: str = "gripper/franka_gripper_conv.stl"):
        self.name = 'gripper'
        self.color = [0, 0, 255]
        self.width = 0.08  # 80mm 
        self.marker_radius = 0.001
        
        if custom:
            self.marker = trimesh.util.concatenate(self._create_custom_gripper())
            self.magnitude = np.linalg.norm(self.marker.center_mass - self.get_grasp_center_point())
            self.gripper_total_length = np.linalg.norm(np.subtract(self._base1[0], self._base1[1])) + \
                                        np.linalg.norm(np.subtract(self._left[0], self._left[1]))
                                        
        else:
            self.marker = trimesh.load(gripper_path)
            self.gripper_total_length = 0.127-0.009-0.009-0.0045#-0.003
            
        self.marker.visual.face_colors = self.color
        self.axis = trimesh.creation.axis(origin_color=[1., 0, 0])
        self.axis.apply_scale(0.07)
        
    def _create_custom_gripper(self):
        y = (self.width/2 + self.marker_radius)
        # franke finger lenght 0.05 = 50: 0.1159999996 - 0.0659999996 = 0.05: -0.01 offset
        self._left = [[0, y, 0.0659999996],  # base
                      [0, y, 0.1159999996 - 0.01]]  # top
        self._right = [[0, -y, 0.0659999996],  # base
                       [0, -y, 0.1159999996 - 0.01]]  # top
        self._base1 = [[0, 0, 0],  # base
                       [0, 0, 0.0659999996]]  # top
        self._base2 = [[0, -y, 0.0659999996],
                       [0, y, 0.0659999996]]
        cfl = trimesh.creation.cylinder(
            radius=self.marker_radius,
            segment=self._left
        )
        cfr = trimesh.creation.cylinder(
            radius=self.marker_radius,
            segment=self._right
        )
        cb1 = trimesh.creation.cylinder(
            radius=self.marker_radius,
            segment=self._base1
        )
        cb2 = trimesh.creation.cylinder(
            radius=self.marker_radius,
            segment=self._base2
        )

        return [cb1, cb2, cfr, cfl]

    def get_grasp_center_point(self):
        right_gripper_top = np.asarray([self._right[1]])
        left_gripper_top = np.asarray([self._left[1]])
        right_gripper_base = np.asarray([self._right[0]])
        left_gripper_base = np.asarray([self._left[0]])

        center_point = np.mean([right_gripper_top, left_gripper_top, right_gripper_base, left_gripper_base], axis=0)
        return center_point

    def transform(self, matrix):
        self.marker.apply_transform(matrix)
        self.axis.apply_transform(matrix)

    def transformT(self, angle=0, translation=[0, 0, 0], direction=[0, 0, 1]):
        rad = np.deg2rad(angle)
        index = np.where(np.array(direction) == 1)[0][0]
        angles = [0, 0, 0]
        angles[index] = rad

        T = trimesh.transformations.compose_matrix(angles=angles, translate=translation)
        self.marker.apply_transform(T)
