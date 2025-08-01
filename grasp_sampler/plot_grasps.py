import trimesh
from gripper import Gripper as g2
import h5py
from scene import Scene
from utlis import *
from scipy.spatial.transform import Rotation as R


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


if __name__ == '__main__':

    scene_v = Scene.from_dict(
        load_yaml("/share/scenes/scene.yaml")
    )
    scene_v2 = Scene.from_dict(
        load_yaml("/share/scenes/scene.yaml")
    )

    names = ["object109505"]
    
    gripper = g2()
    scene_v.add(gripper.marker.copy())
    scene_v.show()
    grasp_all = []
    with h5py.File('grasps/scene_andr20.hdf5', 'r') as f:
        for ii in range(len(names)):
            dset = np.array(f[names[ii]][()])
    
            grasp = []
            for i, (t, t2) in enumerate(dset):
                grasp.append(t)
                grasp.append(t2)
            print(len(grasp))
            grasp_all.append(grasp)

    # grasp_all = [grasp_all[2]]
    for ga in [grasp_all]:
        for idx, g in enumerate(ga):
            print("len: ", len(g))
            for i, (t) in enumerate(g):
                # t = t[0]
                center = t[:3, 3].copy()
                print(t)
                gripper.transform(t)
                
                normal_vector, orientation_vector, approach_vector = t[:3, 0], t[:3, 1], t[:3, 2]
                
                translation = np.eye(4)
                translation[:3, 3:] = np.transpose([(-gripper.gripper_total_length+gripper.marker_radius) * approach_vector])
                gripper.transform(translation)
                
                if i <= 30:
                    #print(scene_v.collision_manager.min_distance_single(gripper.marker))
                    #print(t)
                    r = R.from_matrix(t[:3, :3])
                    #print(np.roll(r.as_quat(), 1))
                    gripper.marker.visual.face_colors = [0, 255, 0]
                    if np.random.rand() < 0.3 and names[idx] != "banana":
                        gripper.marker.visual.face_colors = [255, 0, 0]
                    scene_v.add(gripper.marker.copy())
                    gripper.marker.visual.face_colors = [0, 255, 0]
                    '''
                    p = trimesh.PointCloud([center])
                    scene_v.add(p)
                    f = trimesh.load_path([center, center + 0.1 * orientation_vector])
                    scene_v.add(f)
                    f = trimesh.load_path([center, center - (0.127-0.009-0.0045) * approach_vector])
                    scene_v.add(f)
                    '''
                    #print(center + np.array([0, 0, -0.82436067]))
                scene_v2.add(gripper.marker.copy())
                '''tt = np.dot(translation, t)
                t_inv = trimesh.transformations.inverse_matrix(tt)
                gripper.transform(t_inv)'''
                gripper = g2()
                
                r = R.from_matrix(t[:3, :3])
        
    axis_plane = trimesh.creation.axis(origin_color=[1., 0, 0])
    axis_plane.apply_scale(0.2)

    #scene_v2.show()
    scene_v.show()
    # scene_v.save_image((1920*20, 1080*20))
    # scene_v.save_image((2500, 2500))
