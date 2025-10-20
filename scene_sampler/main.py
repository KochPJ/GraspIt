import os 
import sys
import pynvml
import docker
import threading
import cv2
import imageio
import time
import subprocess
import shutil
from time import sleep
import glob
import numpy as np
from shutil import move
import argparse
import json
from typing import *
from PIL import Image
import fnmatch


def main() -> None:
    threads = {}
    container_count = 0


    #CLI
    while True:
        print("Initializing multi-gpu scene-sampler")
        while True:
            print("0: generate new scenes")
            print("1: add frames to existing scenes")
            mode = input().lower()
            if mode == "1":
                add_frames()
                return
            print("Number of gpus currently available for sampling:")
            print_gpus()
            num_threads = int(input("Enter number of gpus for use:"))
            if num_threads > get_gpu_count():
                print("Invalid option, number of selected gpus > number of available gpus!")
            else:
                print(f"Selected {num_threads} gpus for scene-sampling")
                used_gpus = set([i for i in range(num_threads, get_gpu_count())])
                break
        
        
        asset_path = input("Enter path to assets to be loaded during scene generation (default: /share/):")
        if os.path.exists(asset_path):
            print(asset_path)
        elif asset_path == "":
            asset_path = "/share/"
        else:
            print("Not a valid asset path, defaulting to /share/")
            asset_path = "/share/"

        batch_size = input("Enter batchsize for scene sampling (default: 50):")
        if batch_size == "":
            batch_size = 50
        else:
            batch_size = int(batch_size)
        
        num_scenes = input("Enter number of scenes to be sampled (default: 500):")
        if num_scenes == "":
            num_scenes = 500
        else:
            num_scenes = int(num_scenes)
        
        exces_batch = num_scenes % batch_size
        num_scenes -= num_scenes % batch_size
        batches = num_scenes // batch_size

        print(f"Finished initialization with {num_threads} gpus, {num_scenes} scenes and batchsize of {batch_size}, leading to exces batch of size {exces_batch}")
        proceed = input("Run scene sampler with selected config? [Y/N]").lower()
        if proceed == "y":
            break
    
    #handeling of exces batch in case number of scenes and batchsize are not divisable
    if exces_batch > 0:
        gpu = query_gpu(80.0, used_gpus)
        thread = threading.Thread(target=start_scene_sampler, args=(gpu, container_count, exces_batch))
        threads[container_count] = (thread, gpu)
        thread.start()
        container_count += 1
        used_gpus.add(gpu)


    #main loop for scene generation
    while batches > 0:
        if len(threads.keys()) < num_threads:
            gpu = query_gpu(90.0, used_gpus)
            if gpu is False:
                continue
            print("Selecting gpu {} for thread with id {}".format(gpu, container_count))
            thread = threading.Thread(target=start_scene_sampler, args=(gpu, container_count, batch_size, asset_path))
            threads[container_count] = (thread, gpu)
            thread.start()
            print(used_gpus, threads)
            container_count += 1
            batches -= 1
            used_gpus.add(gpu)
            time.sleep(60)
        to_delete = []
        for key in threads.keys():
            thread, gpu = threads[key]
            if not thread.is_alive():
                to_delete.append(key)
        
        for key in to_delete:
            thread, gpu = threads[key]
            used_gpus.remove(gpu)
            del threads[key]

    #waiting for all remaining containers to finish running
    for key in threads.keys():
        thread, gpu = threads[key]
        thread.join()
    
    clean_new_scenes()

def add_frames():
    while True:
        print("Adding frames to existing scenes, please enter path to dataset:")
        dataset = input("")
        if not os.path.exists(dataset):
            print("Path does not exist, please enter valid path")
            continue
        print(f"adding views to: {dataset}")
        print("0: Replace current frames")
        print("1: Generate additional frames with new poses")
        mode = input()
        mode = "add_images" if mode == "1" else "replace_images"
        
        if mode == "add_images":
            num_views = int(input("Enter number of added images: "))
            asset_path = input("Enter path to assets to be loaded (Defaul: /share/)")
            if asset_path == "":
                asset_path = "/share/"
            gpu = query_gpu(80.0, set())
            thread = threading.Thread(
                target=start_scene_mod,
                args=(gpu, num_views, asset_path, dataset, mode, 0)
            )
            thread.start()
            thread.join()
        else:
            num_views = int(input("Enter number of frames: "))
            asset_path = input("Enter path to assets to be loaded (Defaul: /share/)")
            if asset_path == "":
                asset_path = "/share/"
            gpu = query_gpu(80.0, set())
            thread = threading.Thread(
                target=start_scene_mod,
                args=(gpu, num_views, asset_path, dataset, mode, 0)
            )
            thread.start()
            thread.join()

        break
    
    
    if mode == "add_images":
        add_frames_to_dataset(mode, dataset, num_views)
    else:
        replace_images()

def mod_params(cam_path, final_path):
    (w, h), (fx, fy, cx, cy), V_wc, meters_per_unit, cam_params = load_cam(cam_path)
    cam_params["fx"] = fx
    cam_params["fy"] = fy
    cam_params["cx"] = cx
    cam_params["cy"] = cy
    cam_params["ViewTransformWorldCamera"] = [float(v) for v in V_wc.flatten()]
    with open(final_path, "w+") as f:
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
    shutil.rmtree("dataset")
    os.makedirs("dataset", exist_ok=False)
    index = 0
    for batch in os.listdir("temp"):
        for scene in os.listdir(os.path.join("temp", batch)):
            scene_path = os.path.join("temp", batch, scene)
            path = os.path.join("dataset", "scene_{}".format(index))
            

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

    for file in glob.glob("temp/*"):
        shutil.rmtree(file)
    for file in glob.glob("out/scenes/*"):
        shutil.rmtree(file)
    for file in glob.glob("out/yaml/*"):
        shutil.rmtree(file)

def add_frames_to_dataset(mode, dataset, num_views):
    if mode == "add_images":
        for scene in os.listdir("temp"):
            path = os.path.join(dataset, scene)
            scene_path = os.path.join("temp", scene)
            max_frame = (max([int(item.split("_")[-1]) for item in os.listdir(path) if fnmatch.fnmatch(item, "*_*")]))
            frame_index = max_frame + 1
            end_numbers = []
            temp_numbers = []
            for _ in range(max_frame + 1, max_frame + num_views + 1):
                end_number = "{}".format(_)
                temp_number = "{}".format(_ - (max_frame + 1))
                while len(end_number) < 4:
                    end_number = "0"+ end_number
                while len(temp_number) < 4:
                    temp_number = "0" + temp_number
                end_numbers.append(end_number)
                temp_numbers.append(temp_number)
        
            for end_number, temp_number in zip(end_numbers, temp_numbers):
                cam_params = "camera_params_{}.json"
                depth =  "distance_to_image_plane_{}.npy"
                rgb = "rgb_{}.png"
                mask = "semantic_segmentation_{}.png"
                mask_label = "semantic_segmentation_labels_{}.json"

                frame_path = os.path.join(path, f"frame_{frame_index}")
                os.makedirs(frame_path)

                mod_params(os.path.join(scene_path, cam_params.format(temp_number)), os.path.join(frame_path, cam_params.format(end_number)))
                move(os.path.join(scene_path, rgb.format(temp_number)), os.path.join(frame_path, rgb.format(end_number)))
                move(os.path.join(scene_path, mask.format(temp_number)), os.path.join(frame_path, mask.format(end_number)))
                move(os.path.join(scene_path, mask_label.format(temp_number)), os.path.join(frame_path, mask_label.format(end_number)))

                depth_image = np.load(os.path.join(scene_path, depth.format(temp_number)))
                depth_image[depth_image< 0] = 0
                depth_image[depth_image > 65000/5000] = 0
                depth_image = depth_image*5000

                depth = np.array(depth_image, dtype=np.uint16)
                depth = Image.fromarray(depth)
                depth.save(os.path.join(frame_path, "depth.png"))
                frame_index += 1
    
def replace_images():
    """Cleans up temporary directorys used by containers and generates complete dateset
    """
    index = 0
    for scene in os.listdir("temp"):
        scene_path = os.path.join("temp", scene)
        path = os.path.join("dataset", "scene_{}".format(index))
        

        #computes numbers in format output by Isaac-Sim replicator
        num_frames = len(os.listdir(scene_path)) // 5
        numbers = []
        for _ in range(num_frames):
            number = "{}".format(_)
            while len(number) < 4:
                number = "0" + number
            numbers.append(number)
        
        #setting up dataset directory
        os.makedirs(path, exist_ok=True)
        scene_index = scene.split("_")[0]

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



def start_scene_sampler(gpu: int, id: int, num_scenes: int, asset_path: str):
    """starts Isaac-Sim container with defined config

    Args:
        gpu (int): Index of GPU to run container on
        id (int): Runtime-ID of container for mutex
        num_scenes (int): number of scenes to be generated
        asset_path (str): directory of assets for scene generation
    """
    os.system(f"echo 'Starting Isaac-Sim container: id {id}'")
    os.system(f"./isaac-sim.docker.sh {gpu} {id} {num_scenes} {asset_path} {10}")

def start_scene_mod(gpu: int, num_views: int, asset_path: str, dataset: str, mode: str, id: int):
    """Addes frames to existing datasets created with scene_sampler

    Args:
        gpu (int): Index of GPU to run container on
        num_views (int): number of views to add to dataset
        asset_path (str): directory of assets for scene loading 
        dataset (str): root directory of dataset to be modfified
        mode (str): identifier weather to add or replace images in the dataset
        id (int): Runtime-ID of container for mutex
    """
    os.system(f"echo 'Starting Isaac-sim container")
    os.system(f"./isaac-sim.docker_add_images.sh {gpu} {num_views} {asset_path} {dataset} {mode} {id} {0}")

def print_gpus():
    """prints available GPUs
    """
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    for _ in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(_)
        name = pynvml.nvmlDeviceGetName(handle)
        print(_, name)


def query_gpu(threshold: float, used_gpus: int) -> int:
    """Queries available GPUs and selects GPU to run job

    Args:
        threshold (float): GPU-usage threshold for selection
        used_gpus (int): GPUs currently running scene sampler

    Returns:
        int: Index of GPU selected for job processing
    """
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    for index in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(util.gpu, util.memory)
        if util.gpu < threshold and util.memory < threshold and index not in used_gpus:
            return index
    return False


def get_gpu_count() -> int:
    """Queries number of NVIDIA-GPUs available on current system

    Returns:
        int: number of visible NVIDIA-GPUs
    """
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetCount()


if __name__ == "__main__":
    main()