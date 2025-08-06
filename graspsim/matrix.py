from scipy.spatial.transform import Rotation as R
import numpy as np


class T_Matrix:
    def __init__(self, t: np.ndarray) -> None:
        if t.shape != (4, 4):
            raise ValueError(
                "Passed array is not of the right shape.\n"
                "Expected shape (4, 4) but got array with shape {}.".format(t.shape)
            )

        self._homogeneous_matrix = t
        self._r = R.from_matrix(t[:3, :3])

    def get_homogeneous(self) -> np.array:
        return self._homogeneous_matrix.copy()

    def get_rotation(self, degrees: bool = False) -> np.ndarray:
        return self._r.as_euler("xyz", degrees=degrees)
    
    def get_quat(self) -> np.ndarray:
        return self._r.as_quat()
    
    def get_inv_quat(self) -> np.ndarray:
        return self._r.inv().as_quat()
    
    def get_matrix(self) -> np.ndarray:
        return self._r.as_matrix()

    def get_translation(self) -> np.ndarray:
        return self._homogeneous_matrix[:3, 3].copy()

    def get_normal_vector(self) -> np.array:
        return self.get_matrix()[:3, 0]

    def get_orientation_vector(self) -> np.array:
        return self.get_matrix()[:3, 1]

    def get_approach_vector(self) -> np.array:
        return self.get_matrix()[:3, 2]
    
    def get_inverse_matrix(self):
        return self._r.inv().as_matrix()
