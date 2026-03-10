import numpy as np
import json
import os
import shutil
from shutil import move
from PIL import Image
import glob
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Cleanup-Helper for Scene-Sampler output")
    parser.add_argument("--mode", choices=["clean_dataset", "merge_dataset", "add_grasps"])

    return parser.parse_args()


def mod_params(cam_path, final_path):
    (w, h), (fx, fy, cx, cy), V_wc, meters_per_unit, cam_params = load_cam(cam_path)
    cam_params["fx"] = fx
    cam_params["fy"] = fy
    cam_params["cx"] = cx
    cam_params["cy"] = cy
    cam_params["ViewTransformWorldCamera"] = [float(v) for v in V_wc.flatten()]
    with open(final_path, "w") as f:
        json.dump(cam_params, f)


def load_cam(cam_path):
    with open(cam_path, "r") as f:
        d = json.load(f)
    w, h = d["renderProductResolution"]
    P = np.array(d["cameraProjection"]).reshape(4, 4)
    V = np.array(d["cameraViewTransform"]).reshape(4, 4).T
    ang = np.deg2rad(180.0)
    Rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(ang), -np.sin(ang), 0],
                   [0, np.sin(ang), np.cos(ang), 0],
                   [0, 0, 0, 1]], dtype=np.float64)
    V = Rx @ V

    cameraAperture = d["cameraAperture"]
    cameraApertureOffset = d["cameraApertureOffset"]
    focal_length =  d["cameraFocalLength"]

    fx, fy, cx, cy = parse_intrinsics_from_projection(P, w, h, cameraAperture, cameraApertureOffset, focal_length)
    m_per_unit = float(d.get("metersPerSceneUnit", 1.0))
    return (w, h), (fx, fy, cx, cy), V, m_per_unit, d

def parse_intrinsics_from_projection(P, w, h, cameraAperture, cameraApertureOffset, focal_length):
    fx = P[0, 0] * w/2
    fy = P[1, 1] * h/2
    cx = w/2
    cy = h/2
    return fx, fy, cx, cy


def clean_new_scenes():
    """Cleans up temporary directorys used by containers and generates complete dateset
    """
    if "0_temp" not in os.listdir("temp"):
        for scene in os.listdir("temp"):
                scene_path = os.path.join("temp", scene)
                index = scene.split("_")[-1]
                print(f"cleaning scene: {scene_path}")
                path = os.path.join("/dataset", "scene_{}".format(index))
                

                #computes numbers in format output by Isaac-Sim replicator
                num_frames = len(os.listdir(scene_path)) // 5
                numbers = []
                for _ in range(num_frames):
                    number = "{}".format(_)
                    while len(number) < 4:
                        number = "0" + number
                    numbers.append(number)

                #scene-wise moving of output
                for _index, number in enumerate(numbers):
                    frame_path = os.path.join(path, f"frame_{_index}")
                    os.makedirs(frame_path, exist_ok=True)

                    cam_params = "camera_params_{}.json".format(number)
                    depth =  "distance_to_image_plane_{}.npy".format(number)
                    rgb = "rgb_{}.png".format(number)
                    mask = "semantic_segmentation_{}.png".format(number)
                    mask_label = "semantic_segmentation_labels_{}.json".format(number)

                    mod_params(os.path.join(scene_path, cam_params), os.path.join(frame_path, cam_params))
                    move(os.path.join(scene_path, rgb), os.path.join(frame_path, rgb))
                    move(os.path.join(scene_path, mask), os.path.join(frame_path, mask))
                    move(os.path.join(scene_path, mask_label), os.path.join(frame_path, mask_label))

                    depth_image = np.load(os.path.join(scene_path, depth))
                    depth_image[depth_image< 0] = 0
                    depth_image[depth_image > 65000/5000] = 0
                    depth_image = depth_image*5000

                    depth = np.array(depth_image, dtype=np.uint16)
                    depth = Image.fromarray(depth)
                    depth.save(os.path.join(frame_path, "depth.png"))
                
    else:
        index = 0
        for batch in os.listdir("temp"):
            for scene in os.listdir(os.path.join("temp", batch)):
                scene_path = os.path.join("temp", batch, scene)
                print(f"cleaning scene: {scene_path}")
                path = os.path.join("/dataset", "scene_{}".format(index))
                

                #computes numbers in format output by Isaac-Sim replicator
                num_frames = len(os.listdir(scene_path)) // 5
                numbers = []
                for _ in range(num_frames):
                    number = "{}".format(_)
                    while len(number) < 4:
                        number = "0" + number
                    numbers.append(number)
                
                #setting up dataset directory
                os.makedirs(path, exist_ok=False)
                batch_index = batch.split("_")[0]
                scene_index = scene.split("_")[0]
                yaml_path = f"out/yaml/{batch_index}_yaml/{scene_index}.yaml"
                move(yaml_path, os.path.join(path, "scene.yaml"))

                #scene-wise moving of output
                for _index, number in enumerate(numbers):
                    frame_path = os.path.join(path, f"frame_{_index}")
                    os.makedirs(frame_path, exist_ok=True)

                    cam_params = "camera_params_{}.json".format(number)
                    depth =  "distance_to_image_plane_{}.npy".format(number)
                    rgb = "rgb_{}.png".format(number)
                    mask = "semantic_segmentation_{}.png".format(number)
                    mask_label = "semantic_segmentation_labels_{}.json".format(number)

                    mod_params(os.path.join(scene_path, cam_params), os.path.join(frame_path, cam_params))
                    move(os.path.join(scene_path, rgb), os.path.join(frame_path, rgb))
                    move(os.path.join(scene_path, mask), os.path.join(frame_path, mask))
                    move(os.path.join(scene_path, mask_label), os.path.join(frame_path, mask_label))

                    depth_image = np.load(os.path.join(scene_path, depth))
                    depth_image[depth_image< 0] = 0
                    depth_image[depth_image > 65000/5000] = 0
                    depth_image = depth_image*5000

                    depth = np.array(depth_image, dtype=np.uint16)
                    depth = Image.fromarray(depth)
                    depth.save(os.path.join(frame_path, "depth.png"))
                
                index += 1

    print("==========================")
    print("finished dataset generation, cleaning up...")
    print("==========================")

    #for file in glob.glob("temp/*"):
    #    shutil.rmtree(file)
    #for file in glob.glob("out/scenes/*"):
    #    shutil.rmtree(file)
    #for file in glob.glob("out/yaml/*"):
    #    shutil.rmtree(file)
    
def add_grasps_to_scenes():
    for scene in os.listdir("/dataset"):
        print(scene)

def merge_dataset(source, target):
    max_count = max([element.split("_")[-1] for element in os.listdir(source)])
    print(max_count)

if __name__ == "__main__":
    args = parse_args()

    if args.mode == "clean_dataset":
        clean_new_scenes()
    elif args.mode == "merge_dataset":
        source = input("specify source directory:")
        source = os.path.join("/mnt/4TBSSD/synthetic_data/share", source)

        target = input("Specify target directroy:")
        target = os.path.join("/mnt/4TBSSD/synthetic_data/share", target)

        print(source, target)
    elif args.mode == "add_grasps":
        add_grasps_to_scenes()