import time

import trimesh
from scene import Scene
from typing import List, Optional
from gripper import Gripper
from trimesh.sample import sample_surface, sample_surface_even
import numpy as np
from utlis import *
import os
import h5py


class GraspSampler(object):
    def __init__(self, args, scene: Optional[Scene], scene_path_yml: str):
        """
        """

        self.scene = scene
        self.n_scenes = 1
        self.n_objects = scene.__len__()
        
        self.gripper = Gripper(custom=False)
        self.gripper_custom = Gripper(custom=True)
        
        self.ray_mesh_intersector = None

        self.num_points = args.num_points
        self.max_grasps = args.max_grasps
        self.max_rotations = args.max_rotations
        if self.max_rotations < 1:
            self.max_rotations = 1
        self.friction = args.friction
        self.visualize_single = args.visualize_single
        self.visualize_all = args.visualize_all

        root, filename = os.path.split(scene_path_yml)
        self.storage_path = root
        self.scene_path_yml = scene_path_yml

        if self.max_grasps > self.num_points:
            raise Exception("num_points must be greater than or equal to num_grasps.")

    def sample(self):
        """with multiprocessing
        """

        if self.n_scenes > 1:
            raise NotImplementedError("")
        else:
            self.sample_single()

        return
  
    def sample_single(self):
        """
        """

        start = time.perf_counter()

        grasps = []

        for obj in self.scene.object:
            
            mesh = obj.mesh
            points, face_index = sample_surface(mesh, self.num_points)
            normals = mesh.face_normals[face_index]

            grasps_obj = []

            # without replacement
            idx = list(range(self.num_points))
            sampled_idx = np.random.choice(idx, self.max_grasps, replace=False)

            for i in range(self.max_grasps):

                p1_idx = sampled_idx[i]

                # first contact point
                p1 = points[p1_idx]
                n1 = normals[p1_idx]

                # intersection of the line with the mesh defines the second contact point
                intersections = self._calculate_intersection(mesh, p1, n1)

                # find the furthest intersection point to p1
                idx_max = np.linalg.norm(intersections - p1, axis=1).argmax()

                # find the closest surface point to the intersection point
                po_i = points - intersections[idx_max]
                # ~3x faster faster than "np.linalg.norm(pi, axis=1).argmin()"
                idx_min = np.sqrt(np.einsum('...i,...i', po_i, po_i)).argmin()

                # second contact point
                p2 = points[idx_min]
                n2 = normals[idx_min]
                
                # check whether the object will fit between the gripper
                distance_contact_points = np.linalg.norm(p1 - p2)
                if distance_contact_points < self.gripper.width and distance_contact_points != 0:
                    if self._is_antipodal(p1, p2, n1, n2):
                        # TCP
                        center = np.mean([p1, p2], axis=0)

                        # check whether the TCP is above the table height
                        if center[2] > self.scene.table_height: 
                        
                            # transform gripper align the x-axis of n1
                            trans_mat_x_n1, angle = trimesh.geometry.align_vectors([0, 1, 0], n1, return_angle=True)
                            trans_mat_x_n1[:3, 3:] = np.transpose(np.array([center]))

                            # rotate uniformly around the connecting line
                            # without replacement
                            values = np.linspace(0, 2 * np.pi, 360)
                            angles = np.random.choice(values, self.max_rotations, replace=False)
                            # angles = np.random.uniform(0, 2 *.copy()) np.pi, max_rotations)
                            for angle in angles:
                                trans_mat_n1 = trimesh.transformations.rotation_matrix(angle, n1, center)

                                # total rotation r
                                r = np.dot(trans_mat_n1, trans_mat_x_n1)
                                self.gripper.transform(r)

                                # using the approach vector of n1 to translate the gripper
                                # cm_c_u = self._normalize(gripper.marker.center_mass - center)
                                normal_vector, orientation_vector, approach_vector = r[:3, 0], r[:3, 1], r[:3, 2]
                                
                                translation = np.eye(4)
                                translation[:3, 3:] = np.transpose([(-self.gripper.gripper_total_length+self.gripper.marker_radius) * approach_vector])
                                self.gripper.transform(translation)

                                # total transformation t
                                t = np.dot(np.dot(translation, trans_mat_n1), trans_mat_x_n1)

                                # check whether the gripper is in collision with the mesh
                                if self.scene.min_distance_with(self.gripper.marker) >= 0.00266:
                                    if not self.scene.in_collision_with(self.gripper.marker):
                                        r[:3, 3] = center
                                        rotation_cm_c_u = trimesh.transformations.rotation_matrix(np.pi, approach_vector, center)
                                        r_flipped = np.dot(rotation_cm_c_u, r)
                                        
                                        t_all = np.zeros((1, 2, 4, 4))
                                        t_all[0][0] = r
                                        t_all[0][1] = r_flipped
                                        grasps_obj.append(t_all)
                                
                                        if self.visualize_single:
                                            name_gripper = self.scene.add(self.gripper.marker.copy())

                                            p = trimesh.PointCloud([center])
                                            name_p1 = self.scene.add(p)
                                            f = trimesh.load_path([p1, p1 + 0.01 * n1])
                                            name_p2 = self.scene.add(f)
                                            f = trimesh.load_path([p2, p2 + 0.01 * n2])
                                            name_p3 = self.scene.add(f)

                                            self.scene.show()
                                            
                                            self.scene.delete(name_gripper)
                                            self.scene.delete(name_p1)
                                            self.scene.delete(name_p2)
                                            self.scene.delete(name_p3)

                                # set the gripper position to the initial position
                                t_inv = self._inverse(t)
                                self.gripper.transform(t_inv)
                                
            print(len(grasps_obj))
            grasps.append(grasps_obj)
        
        end = time.perf_counter()
        real_time = round(end - start, 3)

        n_grasps = 0
        for i in range(len(grasps)):
            n_grasps += len(grasps[i])
            
        print('{} grasps were sampled in {} s.'.format(n_grasps, real_time))
        self._save_dataset(grasps)

        if self.visualize_all:
            self._visualize_all_grasps(grasps)

        return

    def _is_antipodal(self, p1, p2, n1, n2) -> bool:
        """
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=246063
        """

        cf = np.cos(np.arctan(self.friction))

        diff = p1 - p2
        p_unit = self._normalize(diff)
        c = np.dot(n1, p_unit)

        if c > cf:
            diff = p2 - p1
            p_unit = self._normalize(diff)
            c = np.dot(n2, p_unit)

            if c > cf:
                return True

        return False

    def _calculate_intersection(self, mesh: trimesh.Trimesh, p1, n1):
        """
        """
        _, _, intersections = mesh.ray.intersects_id(ray_origins=[p1],
                                                     ray_directions=[-n1],
                                                     multiple_hits=True,
                                                     return_locations=True,
                                                     max_hits=1000)
        
        return intersections

    def _normalize(self, v) -> np.ndarray:
        """
        """
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def _inverse(self, matrix) -> np.ndarray:
        """
        """
        return trimesh.transformations.inverse_matrix(matrix)

    def _visualize_all_grasps(self, grasps: List[List[np.ndarray]]) -> None:
        """
        """
        n_grasps = 2
        for grasps_obj in grasps:
            for t_all in grasps_obj[:n_grasps]:
                t = t_all[0][1] 
                
                self.gripper_custom.transform(t)
                
                approach_vector = t[:3, 2]
                
                translation = np.eye(4)
                translation[:3, 3:] = np.transpose([(-self.gripper_custom.gripper_total_length+self.gripper_custom.marker_radius) * approach_vector])
                self.gripper_custom.transform(translation)
                
                _ = self.scene.add(self.gripper_custom.marker.copy())
                
                # set the gripper position to the initial position
                t = np.dot(translation, t)
                t_inv = self._inverse(t)
                self.gripper_custom.transform(t_inv)
                
        self.scene.show()  

    def _save_dataset(self, grasps) -> None:
        """
        """
        prepared_grasps = []
        for g in grasps:
            t_all = np.zeros((len(g), 2, 4, 4))
            for i, (t) in enumerate(g):
                t_all[i] = t

            prepared_grasps.append(t_all)
        
        with h5py.File(os.path.join(self.storage_path, "grasps" + ".hdf5"), 'w') as f:
            for i in range(self.n_objects):
                print(prepared_grasps[i].shape)
                f.create_dataset(self.scene.object[i].name, data=prepared_grasps[i])

        with open(self.scene_path_yml, 'r') as yaml_file:
            config = yaml.load(yaml_file, yaml.Loader)

        config['grasp_path'] = os.path.join(self.storage_path, "grasps" + ".hdf5")
        with open(self.scene_path_yml, 'w') as yaml_file:
            config = yaml.dump(config, yaml_file, default_flow_style=None, Dumper=yaml.Dumper)

        print("Dataset was saved successfully.")
 
        return
