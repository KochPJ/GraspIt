import time
import logging
import os
from typing import List, Optional

import h5py
import numpy as np
import trimesh
import yaml
from trimesh.sample import sample_surface

from scene import Scene
from gripper import Gripper

logger = logging.getLogger(__name__)

# Constants
MIN_CLEARANCE = 0.00266
ROTATION_RESOLUTION = 360
VISUALIZATION_GRASPS = 10
GRASP_FILENAME = "grasps.hdf5"


class GraspSampler:
    """Samples antipodal grasps on object surfaces within a scene."""

    def __init__(self, args, scene: Scene, scene_path_yml: str):
        """
        Initializes the GraspSampler.

        Args:
            args: Configuration arguments with num_points, max_grasps, max_rotations, friction, etc.
            scene: The scene containing the objects.
            scene_path_yml: Path to the YAML configuration file of the scene.
        """
        if scene is None:
            raise ValueError("scene must not be None.")

        self.scene = scene
        self.n_objects = len(scene)

        self.gripper = Gripper(custom=False)
        self.gripper_custom = Gripper(custom=True)

        self.num_points = args.num_points
        self.max_grasps = args.max_grasps
        self.max_rotations = max(args.max_rotations, 1)
        self.friction = args.friction
        self.visualize_single = args.visualize_single
        self.visualize_all = args.visualize_all

        root, _ = os.path.split(scene_path_yml)
        self.storage_path = root
        self.scene_path_yml = scene_path_yml

        if self.max_grasps > self.num_points:
            raise ValueError("num_points must be greater than or equal to max_grasps.")

    def sample(self) -> List[List[np.ndarray]]:
        """
        Samples grasps for all objects in the scene.

        Returns:
            List of grasp transformations per object.
        """
        start = time.perf_counter()
        grasps = []

        for obj_idx, obj in enumerate(self.scene.objects):
            grasps_obj = self._sample_object(obj, obj_idx)
            logger.info(f"{len(grasps_obj)} grasps for object '{obj.name}'")
            print(f"{len(grasps_obj)} grasps for object '{obj.name}'")
            grasps.append(grasps_obj)

        elapsed = time.perf_counter() - start
        n_grasps = sum(len(g) for g in grasps)
        logger.info(f"{n_grasps} grasps were sampled in {elapsed:.3f} s.")
        print(f"{n_grasps} grasps were sampled in {elapsed:.3f} s.")

        self._save_dataset(grasps)

        if self.visualize_all:
            self._visualize_all_grasps(grasps)

        return grasps

    def _sample_object(self, obj, obj_idx: int) -> List[np.ndarray]:
        """
        Samples grasps for a single object.

        Args:
            obj: The object to grasp.
            obj_idx: Index of the object in the scene.

        Returns:
            List of valid grasp transformations.
        """
        mesh = obj.mesh
        points, face_index = sample_surface(mesh, self.num_points)
        normals = mesh.face_normals[face_index]

        grasps_obj = []
        sampled_idx = np.random.choice(self.num_points, self.max_grasps, replace=False)

        for i in range(self.max_grasps):
            p1 = points[sampled_idx[i]]
            n1 = normals[sampled_idx[i]]

            p2, n2 = self._find_second_contact_point(mesh, points, normals, p1, n1)
            if p2 is None:
                continue

            grasp_candidates = self._evaluate_grasp_candidate(p1, p2, n1, n2, obj_idx)
            grasps_obj.extend(grasp_candidates)

        return grasps_obj

    def _find_second_contact_point(
        self,
        mesh: trimesh.Trimesh,
        points: np.ndarray,
        normals: np.ndarray,
        p1: np.ndarray,
        n1: np.ndarray,
    ) -> tuple:
        """
        Finds the second contact point based on a ray cast along the inward normal.

        Returns:
            (p2, n2) or (None, None) if no intersection was found.
        """
        intersections = self._calculate_intersection(mesh, p1, n1)
        if len(intersections) == 0:
            return None, None

        # Furthest intersection point from p1
        idx_max = np.linalg.norm(intersections - p1, axis=1).argmax()

        # Closest surface point to the intersection
        diff = points - intersections[idx_max]
        idx_min = np.sqrt(np.einsum('...i,...i', diff, diff)).argmin()

        p2 = points[idx_min]
        n2 = normals[idx_min]
        return p2, n2

    def _evaluate_grasp_candidate(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        n1: np.ndarray,
        n2: np.ndarray,
        obj_idx: int,
    ) -> List[np.ndarray]:
        """
        Evaluates a grasp candidate and returns valid transformations.

        Args:
            p1, p2: Contact points.
            n1, n2: Normals at the contact points.
            obj_idx: Index of the current object.

        Returns:
            List of valid grasp transformations.
        """
        distance = np.linalg.norm(p1 - p2)
        if distance == 0 or distance >= self.gripper.width:
            return []

        if not self._is_antipodal(p1, p2, n1, n2):
            return []

        center = np.mean([p1, p2], axis=0)
        if center[2] <= self.scene.table_height:
            return []

        return self._try_rotations(p1, p2, n1, n2, center, obj_idx)

    def _try_rotations(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        n1: np.ndarray,
        n2: np.ndarray,
        center: np.ndarray,
        obj_idx: int,
    ) -> List[np.ndarray]:
        """
        Tests different gripper rotations around the connecting line between contact points.

        Args:
            p1, p2: Contact points.
            n1, n2: Normals.
            center: TCP (Tool Center Point).
            obj_idx: Index of the current object.

        Returns:
            List of valid grasp transformations.
        """
        results = []

        trans_mat_x_n1, _ = trimesh.geometry.align_vectors(
            [0, 1, 0], n1, return_angle=True
        )
        trans_mat_x_n1[:3, 3] = center

        # Random rotations around the connecting line
        values = np.linspace(0, 2 * np.pi, ROTATION_RESOLUTION)
        angles = np.random.choice(values, self.max_rotations, replace=False)

        for rot_angle in angles:
            grasp = self._test_single_rotation(
                rot_angle, n1, n2, center, p1, p2, trans_mat_x_n1, obj_idx
            )
            if grasp is not None:
                results.append(grasp)

        return results

    def _test_single_rotation(
        self,
        rot_angle: float,
        n1: np.ndarray,
        n2: np.ndarray,
        center: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        trans_mat_x_n1: np.ndarray,
        obj_idx: int,
    ) -> Optional[np.ndarray]:
        """
        Tests a single rotation and returns the transformation if valid.

        Args:
            rot_angle: Rotation angle around the normal.
            n1, n2: Normals.
            center: TCP.
            p1, p2: Contact points.
            trans_mat_x_n1: Initial alignment transformation.
            obj_idx: Index of the current object.

        Returns:
            Grasp transformation (shape: (1, 2, 4, 4)) or None.
        """
        trans_mat_n1 = trimesh.transformations.rotation_matrix(rot_angle, n1, center)
        r = np.dot(trans_mat_n1, trans_mat_x_n1)
        self.gripper.transform(r)

        approach_vector = r[:3, 2]
        translation = np.eye(4)
        offset = (-self.gripper.gripper_total_length + self.gripper.marker_radius) * approach_vector
        translation[:3, 3] = offset
        self.gripper.transform(translation)

        # Total transformation
        t = np.dot(np.dot(translation, trans_mat_n1), trans_mat_x_n1)

        result = None
        if self._is_grasp_valid():
            r[:3, 3] = center
            rotation_flip = trimesh.transformations.rotation_matrix(
                np.pi, approach_vector, center
            )
            r_flipped = np.dot(rotation_flip, r)

            t_all = np.zeros((1, 2, 4, 4))
            t_all[0, 0] = r
            t_all[0, 1] = r_flipped
            result = t_all

            if self.visualize_single:
                self._visualize_single_grasp(center, p1, p2, n1, n2, obj_idx)

        # Reset gripper to initial position
        t_inv = trimesh.transformations.inverse_matrix(t)
        self.gripper.transform(t_inv)

        return result

    def _is_grasp_valid(self) -> bool:
        """
        Checks whether the gripper is placed collision-free.

        Returns:
            True if the grasp is valid.
        """
        if self.scene.min_distance_to(self.gripper.marker) < MIN_CLEARANCE:
            return False
        if self.scene.in_collision_with(self.gripper.marker):
            return False
        return True

    def _visualize_single_grasp(
        self,
        center: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        n1: np.ndarray,
        n2: np.ndarray,
        obj_idx: int,
    ) -> None:
        """
        Visualizes a single grasp with contact points and normals.

        Args:
            center: TCP.
            p1, p2: Contact points.
            n1, n2: Normals.
            obj_idx: Index of the current object (parent_id).
        """
        self.scene.add(self.gripper.marker.copy(), obj_idx)
        self.scene.add(trimesh.PointCloud([center]), obj_idx)
        self.scene.add(trimesh.load_path([p1, p1 + 0.01 * n1]), obj_idx)
        self.scene.add(trimesh.load_path([p2, p2 + 0.01 * n2]), obj_idx)

        self.scene.show()

        self.scene.delete(obj_idx)

    def _is_antipodal(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        n1: np.ndarray,
        n2: np.ndarray,
    ) -> bool:
        """
        Checks the antipodal condition based on the friction coefficient.

        Reference: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=246063
        """
        cf = np.cos(np.arctan(self.friction))

        p_unit_12 = self._normalize(p1 - p2)
        if np.dot(n1, p_unit_12) <= cf:
            return False

        p_unit_21 = self._normalize(p2 - p1)
        if np.dot(n2, p_unit_21) <= cf:
            return False

        return True

    def _calculate_intersection(
        self,
        mesh: trimesh.Trimesh,
        p1: np.ndarray,
        n1: np.ndarray,
    ) -> np.ndarray:
        """
        Computes intersections of a ray along -n1 with the mesh.

        Returns:
            Array of intersection points.
        """
        _, _, intersections = mesh.ray.intersects_id(
            ray_origins=[p1],
            ray_directions=[-n1],
            multiple_hits=True,
            return_locations=True,
            max_hits=1000,
        )
        return intersections

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        """Normalizes a vector. Returns zero vector if norm is 0."""
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def _visualize_all_grasps(self, grasps: List[List[np.ndarray]]) -> None:
        """Visualizes a limited number of grasps per object."""
        for obj_idx, grasps_obj in enumerate(grasps):
            for t_all in grasps_obj[:VISUALIZATION_GRASPS]:
                t = t_all[0, 1]

                self.gripper_custom.transform(t)

                approach_vector = t[:3, 2]
                translation = np.eye(4)
                offset = (-self.gripper_custom.gripper_total_length + self.gripper_custom.marker_radius) * approach_vector
                translation[:3, 3] = offset
                self.gripper_custom.transform(translation)

                self.scene.add(self.gripper_custom.marker.copy(), obj_idx)

                # Reset gripper to initial position
                total_t = np.dot(translation, t)
                t_inv = trimesh.transformations.inverse_matrix(total_t)
                self.gripper_custom.transform(t_inv)

        self.scene.show()

    def _save_dataset(self, grasps: List[List[np.ndarray]]) -> None:
        """
        Saves the grasps as an HDF5 file and updates the YAML configuration.

        Args:
            grasps: List of grasp transformations per object.
        """
        grasp_path = os.path.join(self.storage_path, GRASP_FILENAME)

        with h5py.File(grasp_path, 'w') as f:
            for i, grasps_obj in enumerate(grasps):
                if len(grasps_obj) == 0:
                    t_all = np.zeros((0, 2, 4, 4))
                else:
                    t_all = np.concatenate(grasps_obj, axis=0)
                logger.info(f"Object '{self.scene.objects[i].name}': shape {t_all.shape}")
                print(f"Object '{self.scene.objects[i].name}': shape {t_all.shape}")
                f.create_dataset(self.scene.objects[i].name, data=t_all)

        # Update YAML configuration
        with open(self.scene_path_yml, 'r') as yaml_file:
            config = yaml.load(yaml_file, Loader=yaml.Loader)

        config['grasp_path'] = grasp_path

        with open(self.scene_path_yml, 'w') as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=None, Dumper=yaml.Dumper)

        logger.info("Dataset was saved successfully.")
        print("Dataset was saved successfully.")