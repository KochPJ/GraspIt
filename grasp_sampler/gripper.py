import numpy as np
import trimesh

# Constants
DEFAULT_GRIPPER_PATH = "gripper/franka_gripper_conv.stl"
DEFAULT_GRIPPER_COLOR = [0, 0, 255]
GRIPPER_WIDTH = 0.08  # 80mm
MARKER_RADIUS = 0.001
AXIS_SCALE = 0.07

# Franka Gripper dimensions
FINGER_LENGTH = 0.05
BASE_HEIGHT = 0.0659999996
GRIPPER_TOTAL_LENGTH_OFFSET = 0.127 - 0.009 - 0.009 - 0.0045


class Gripper:
    """Represents a gripper (Franka Hand) as a mesh for collision checking."""

    def __init__(self, custom: bool = False, gripper_path: str = DEFAULT_GRIPPER_PATH):
        """
        Initializes the gripper.

        Args:
            custom: If True, a simplified cylinder-based gripper is created.
                    If False, the STL mesh is loaded.
            gripper_path: Path to the gripper STL file (Franka Hand) (only used when custom=False).
        """
        self.name = "gripper"
        self.color = DEFAULT_GRIPPER_COLOR
        self.width = GRIPPER_WIDTH
        self.marker_radius = MARKER_RADIUS

        if custom:
            self.marker = trimesh.util.concatenate(self._create_custom_gripper())
            self.magnitude = np.linalg.norm(
                self.marker.center_mass - self.get_grasp_center_point()
            )
            base_length = np.linalg.norm(np.subtract(self._base1[0], self._base1[1]))
            finger_length = np.linalg.norm(np.subtract(self._left[0], self._left[1]))
            self.gripper_total_length = base_length + finger_length
        else:
            self.marker = trimesh.load(gripper_path)
            self.gripper_total_length = GRIPPER_TOTAL_LENGTH_OFFSET

        self.marker.visual.face_colors = self.color
        self.axis = trimesh.creation.axis(origin_color=[1.0, 0, 0])
        self.axis.apply_scale(AXIS_SCALE)

    def _create_custom_gripper(self) -> list:
        """
        Creates a simplified gripper from cylinders.

        Returns:
            List of trimesh cylinders (base1, base2, right finger, left finger).
        """
        y_offset = self.width / 2 + self.marker_radius

        self._left = [
            [0, y_offset, BASE_HEIGHT],
            [0, y_offset, BASE_HEIGHT + FINGER_LENGTH],
        ]
        self._right = [
            [0, -y_offset, BASE_HEIGHT],
            [0, -y_offset, BASE_HEIGHT + FINGER_LENGTH],
        ]
        self._base1 = [
            [0, 0, 0],
            [0, 0, BASE_HEIGHT],
        ]
        self._base2 = [
            [0, -y_offset, BASE_HEIGHT],
            [0, y_offset, BASE_HEIGHT],
        ]

        cfl = trimesh.creation.cylinder(radius=self.marker_radius, segment=self._left)
        cfr = trimesh.creation.cylinder(radius=self.marker_radius, segment=self._right)
        cb1 = trimesh.creation.cylinder(radius=self.marker_radius, segment=self._base1)
        cb2 = trimesh.creation.cylinder(radius=self.marker_radius, segment=self._base2)

        return [cb1, cb2, cfr, cfl]

    def get_grasp_center_point(self) -> np.ndarray:
        """
        Computes the center point of the grasp area.

        Returns:
            Center point between the fingertips and finger bases.
        """
        points = [
            self._right[1],
            self._left[1],
            self._right[0],
            self._left[0],
        ]
        center_point = np.mean(points, axis=0)
        return center_point

    def transform(self, matrix: np.ndarray) -> None:
        """
        Applies a transformation matrix to the gripper and its axis.

        Args:
            matrix: 4x4 homogeneous transformation matrix.
        """
        self.marker.apply_transform(matrix)
        self.axis.apply_transform(matrix)