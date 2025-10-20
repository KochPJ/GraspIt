import argparse


from omni.isaac.kit import SimulationApp
CONFIG = {"renderer": "RayTracedLighting", "headless": True}
simulation_app = SimulationApp(CONFIG)

import omni
import omni.isaac.core.utils.prims as prims_utils
import carb.settings
from omni.isaac.core import World
from pxr import Gf
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import get_current_stage, open_stage
from viewer import Viewer
from scene import Scene
from yaml import safe_load
from time import sleep
from math import pi, sin, cos


from pxr import Usd, UsdGeom, Sdf
from utils import get_current_stage
from typing import List, Tuple
from PIL import Image
import os
import imageio
import numpy as np
from shutil import move, rmtree
from datetime import datetime
import omni.replicator.core as rep


nucleus_server = get_assets_root_path()
print(nucleus_server)

private_assets = "omniverse://localhost/Assets/tables_usd"