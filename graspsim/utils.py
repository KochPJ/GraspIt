from omni.isaac.core.utils.nucleus import get_assets_root_path

from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
)


from pxr import Usd
import omni.usd

from contextlib import contextmanager

import os
import re


def get_current_stage() -> Usd.Stage:
    return omni.usd.get_context().get_stage()

def get_table_usd():
    nucleus_server = get_assets_root_path()
    asset_folder = nucleus_server + "/Isaac/Samples/Examples/FrankaNutBolt/"
    table_usd = asset_folder + "SubUSDs/Shop_Table/Shop_Table.usd"
    return table_usd

def get_robot_cfg():
    robot_cfg_path = get_robot_configs_path()
    print(robot_cfg_path)
    robot_cfg = load_yaml(join_path(robot_cfg_path, "franka.yml"))["robot_cfg"]
    return robot_cfg

@contextmanager
def _ignore_torch_cuda_oom():
    """
    A context which ignores CUDA OOM exception from pytorch.
    """
    try:
        yield
    except RuntimeError as e:
        # NOTE: the string may change?
        if "CUDA out of memory. " in str(e):
            pass
        else:
            raise

def get_scene_paths(base_directory: str, indices: str = "") -> list:
    """
    Returns paths to 'scene.yaml' files in the base_directory.
    Indices can be e.g. '0-5' (range) or '1,3,5' (specific indices).
    """
    scene_paths = [
        os.path.join(root, 'scene.yaml')
        for root, dirs, files in os.walk(base_directory)
        if 'scene.yaml' in files
    ]

    if indices:
        # Range (e.g. 1-5)
        match = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", indices)
        if match:
            start, end = map(int, match.groups())
            return scene_paths[start:end]
        # Specific indices (e.g. 1,3,5)
        elif re.match(r"^\s*\d+(?:\s*,\s*\d+)*\s*$", indices):
            idx_list = [int(i.strip()) for i in indices.split(",")]
            return [scene_paths[i] for i in idx_list if 0 <= i < len(scene_paths)]
        else:
            raise ValueError("Invalid format. Allowed: 'start-end' or 'i1,i2,...'.")
    else:
        return scene_paths