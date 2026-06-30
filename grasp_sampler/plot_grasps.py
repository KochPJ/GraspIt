import trimesh
from gripper import Gripper
import random
import h5py
from scene import Scene
from utlis import *


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def inverse(matrix) -> np.ndarray:
        """
        """
        return trimesh.transformations.inverse_matrix(matrix)


if __name__ == '__main__':
    scene_path = "scenes/scene_0/scene.yaml"
    yaml_params = load_yaml(scene_path)

    scene = Scene.from_dict(yaml_params)
    n_obj = len(scene)
    
    gripper = Gripper(custom=True)
    
    grasp_all = []
    with h5py.File(yaml_params["grasp_path"], 'r') as f:
        for j in range(n_obj):
            dset = np.array(f[scene.objects[j].name][()])
    
            grasp = []
            for i, (t, t2) in enumerate(dset):
                grasp.append(t)
                grasp.append(t2)
            random.shuffle(grasp)
            grasp_all.append(grasp)
    
    for id, ga in enumerate(grasp_all):
        for i, t in enumerate(ga):
            center = t[:3, 3].copy()
            gripper.transform(t)
            
            approach_vector =  t[:3, 2]
            
            translation = np.eye(4)
            translation[:3, 3:] = np.transpose([(-gripper.gripper_total_length+gripper.marker_radius) * approach_vector])
            gripper.transform(translation)
            
            if i <= 10:
                gripper.marker.visual.face_colors = [0, 255, 0]
                scene.add(gripper.marker.copy(), id)
                
            # set the gripper position to the initial position
            t = np.dot(translation, t)
            t_inv = inverse(t)
            gripper.transform(t_inv)
            
    axis_plane = trimesh.creation.axis(origin_color=[1., 0, 0])
    axis_plane.apply_scale(0.2)

    scene.show()
