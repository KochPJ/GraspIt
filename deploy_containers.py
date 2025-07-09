import os 
import sys
import pynvml
import docker
import threading
import cv2
import imageio
import subprocess
import shutil
from time import sleep
import glob
import numpy as np
from shutil import move
import argparse
from typing import *


def parse_args():
    parser = argparse.ArgumentParser(description="Commandlinetool for large scale scene generation")
    parser.add_argument("--num_scenes", default=500, type=int)

    return parser.parse_args()


def main() -> None:
    threads = {}
    container_count = 0
    args = parse_args()
    num_scenes = args.num_scenes
    batches = num_scenes // 2
    num_threads = get_gpu_count()
    used_gpus = set()

    while batches > 0:
        if len(threads.keys()) < num_threads:
            gpu = query_gpu(80.0, used_gpus)
            if gpu is False:
                continue
            print("Selecting gpu {} for thread with id {}".format(gpu, container_count))
            thread = threading.Thread(target=start_container, args=(gpu, container_count))
            threads[container_count] = (thread, gpu)
            thread.start()
            print(used_gpus, threads)
            container_count += 1
            batches -= 1
            used_gpus.add(gpu)
        to_delete = []
        for key in threads.keys():
            thread, gpu = threads[key]
            if not thread.is_alive():
                to_delete.append(key)
        
        for key in to_delete:
            thread, gpu = threads[key]
            used_gpus.remove(gpu)
            del threads[key]
    
    for key in threads.keys():
        thread, gpu = threads[key]
        thread.join()
    
    clean_dataset()



def clean_dataset():
    index = 0
    for batch in os.listdir("temp"):
        for scene in os.listdir(os.path.join("temp", batch)):
            scene_path = os.path.join("temp", batch, scene)
            path = os.path.join("dataset", "scene_{}".format(index))
            

            num_frames = len(os.listdir(scene_path)) // 5
            numbers = []
            for _ in range(num_frames):
                number = "{}".format(_)
                while len(number) < 4:
                    number = "0" + number
                numbers.append(number)
            
            os.makedirs(path, exist_ok=False)
            batch_index = batch.split("_")[0]
            scene_index = scene.split("_")[0]
            yaml_path = f"out/yaml/{batch_index}_yaml/{scene_index}.yaml"
            move(yaml_path, os.path.join(path, "scene.yaml"))

            for _index, number in enumerate(numbers):
                frame_path = os.path.join(path, f"frame_{_index}")
                os.makedirs(frame_path, exist_ok=True)

                cam_params = "camera_params_{}.json".format(number)
                depth =  "distance_to_camera_{}.npy".format(number)
                rgb = "rgb_{}.png".format(number)
                mask = "semantic_segmentation_{}.png".format(number)
                mask_label = "semantic_segmentation_labels_{}.json".format(number)

                move(os.path.join(scene_path, cam_params), os.path.join(frame_path, cam_params))
                move(os.path.join(scene_path, rgb), os.path.join(frame_path, rgb))
                move(os.path.join(scene_path, mask), os.path.join(frame_path, mask))
                move(os.path.join(scene_path, mask_label), os.path.join(frame_path, mask_label))

                depth_image = np.load(os.path.join(scene_path, depth))
                comp_depth = (65535*(depth_image - depth_image.min())/np.ptp(depth_image)).astype(np.uint16)
                imageio.imwrite(os.path.join(frame_path, "depth.png"), comp_depth)  
            
            index += 1

    print("==========================")
    print("finished dataset generation, cleaning up...")
    print("==========================")

    for file in glob.glob("temp/*"):
        shutil.rmtree(file)
    for file in glob.glob("out/scenes/*"):
        shutil.rmtree(file)
    for file in glob.glob("out/yaml/*"):
        shutil.rmtree(filetouch)
        

def echo(id):
    os.system(f"echo 'Starting Thread id {id}'")
    sleep(5)


def start_container(gpu, id):
    os.system(f"echo 'Starting Isaac-Sim container: id {id}'")
    os.system(f"./isaac-sim.docker.sh {gpu} {id}")


def query_gpu(threshold, used_gpus) -> int:
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
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetCount()


if __name__ == "__main__":
    main()