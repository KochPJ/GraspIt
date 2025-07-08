import os 
import sys
import pynvml
import docker
import threading
import subprocess
from time import sleep
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
    batches = num_scenes // 5
    num_threads = get_gpu_count()
    print(batches)

    while batches > 0:
        if len(threads.keys()) < num_threads:
            gpu = query_gpu(80.0)
            if gpu is False:
                continue
            print("Selecting gpu {} for thread with id {}".format(gpu, container_count))
            thread = threading.Thread(target=start_container, args=(gpu, container_count))
            threads[container_count] = thread
            thread.start()
            container_count += 1
            batches -= 1
        to_delete = []
        for key in threads.keys():
            thread = threads[key]
            if not thread.is_alive():
                to_delete.append(key)
        
        for key in to_delete:
            del threads[key]


def echo(id):
    os.system(f"echo 'Starting Thread id {id}'")
    sleep(5)


def start_container(gpu, id):
    os.system(f"echo 'Starting Isaac-Sim container: id {id}'")
    os.system(f"./isaac-sim.docker.sh {gpu} {id}")


def query_gpu(threshold) -> int:
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    for index in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(util.gpu, util.memory)
        if util.gpu < threshold and util.memory < threshold:
            return index
    return False


def get_gpu_count() -> int:
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetCount()


if __name__ == "__main__":
    main()