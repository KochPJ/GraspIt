import argparse

COUNT = 0

def parse_args():
    parser = argparse.ArgumentParser(description="A NVIDIA Ominverse implementation to simulate random tabletops")

    parser.add_argument("--dataset", default="/dataset")
    parser.add_argument("--num_views", default=0)
    parser.add_argument("--container_index", default=0)
    parser.add_argument("--use_poses", action="store_true")

    return parser.parse_args()

args = parse_args()

from omni.isaac.kit import SimulationApp
CONFIG = {"renderer": "RayTracedLighting", "headless": False}
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

def main(scene_path, scene_index):
    world = World()
    root_path = "/World/Workstation"

    print(get_assets_root_path())

    with open(os.path.join(args.dataset, f"scene_{scene_index}", "scene.yaml")) as f:
        data_dict = safe_load(f)

    scene = Scene.from_dataset(data_dict, root_path)
    viewer = Viewer(world=world, scene=scene, root_path=root_path, mode="add_images")
    world.reset()
    viewer.post_reset()
    for _ in range(14):
        world.render()
    while simulation_app.is_running():
        viewer.step_time += 1.0/60.0

        if world.is_stopped():
            print("break")
            break

        if not world.is_playing():
            world.step(render=True)
            continue
        if viewer.step_time > 1:
            break

    stage = get_current_stage()
    stage.Export(scene_path)
    print("======= exiting Simulation =========")
    return viewer, world


def reset_scene(args, viewer, world):
    for _object in viewer.objects:
        prims_utils.delete_prim(_object.prim_path)
    prims_utils.delete_prim("/World/Workstation_0/table")
    prims_utils.delete_prim("/World/Workstation_0")
    prims_utils.delete_prim("/World")
    world.reset()

    for _ in range(100):
        world.render()


def generate_data(args, temp_path, scene_path):
    print(f"====== generating synthetic dataset with {args.num_views} views =======")

    settings = carb.settings.get_settings()
    settings.set("rtx/ambientOcclusion/enabled", True)

    open_stage(scene_path)
    stage = get_current_stage()

    for i in range(100):
        if i % 10 == 0:
            print(f"updating SimulationApp: {i}")
        simulation_app.update()
    focus_distance = 4

    camera = rep.create.camera(
        position=(2, 2, 2),
        look_at=(0, 0, 0),
        focus_distance=focus_distance
    )
    ground_plane = rep.create.plane(
        position=(0, 0, 0.1),
        scale=20,
        semantics=[("class", "plane")],
        rotation=(0, 0, 0)
    )
    dome_light = rep.create.light(
        light_type="dome",
        temperature=6500,
        intensity=1000,
        rotation=(0, 0, -90),
        position=(0, 0, 5)
    )
    distance_light = rep.create.light(
        light_type="distant",
        temperature=6500,
        intensity=1000,
        look_at=(0, 0, 0.82)
    )
    distance_light_1 = rep.create.light(
        light_type="distant",
        temperature=6500,
        intensity=1000,
        look_at=(0, 0, 0.82),
        position=(5, 0, 0)
    )
    distance_light_2 = rep.create.light(
        light_type="distant",
        temperature=6500,
        intensity=1000,
        look_at=(0, 0, 0.82),
        position=(-5, 0, 5)
    )

    prims: List[Usd.Prim] = [x for x in stage.Traverse() if x.IsA(UsdGeom.Mesh)]
    scene_objects = [str(prim.GetPath()) for prim in prims]
    scene_names = [element.split("/")[3] for element in scene_objects]
    prims = [(rep.get.prims(path_pattern=name), name) for name in scene_names if "table" not in name and 'Enviroment' not in name and "Plane" not in name]
    table = [rep.get.prims(path_pattern=name) for name in scene_names if "table" in name][0]
    
    print(prims)
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=f"/home/sersandr/OptiSim/scene_sampler/{temp_path}",
        rgb=True,
        semantic_segmentation=True,
        distance_to_image_plane=True,
        camera_params=True
    )
    render_product = rep.create.render_product(camera, (1920, 1080))
    writer.attach([render_product])
        
    num_views = 10
    textures = [os.path.join("/share/textures", item) for item in os.listdir("/share/textures")]

    with rep.trigger.on_frame(max_execs=num_views, rt_subframes=80):
        with table:
            rep.modify.semantics([('class', 'table')])
        with camera:
            rep.modify.pose(
                position=rep.distribution.uniform((-2, -2, 2), (2, 2, 2)),
                look_at=rep.distribution.uniform((-0.2, -0.2, 0.82), (0.2, 0.2, 0.82))
            )
        for item in prims:
            prim, name = item
            with prim:
                rep.randomizer.materials(
                    rep.create.material_omnipbr(
                        diffuse_texture=rep.distribution.choice(textures),
                        project_uvw=True,
                        roughness=rep.distribution.uniform(0.2, 0.8),
                        metallic=rep.distribution.uniform(0.0, 0.2)
                    )
                )
            rep.modify.semantics([('class', f'{name}')])        
    rep.orchestrator.run()
    simulation_app.update()

def generate_data_from_poses(args, temp_path, scene_path, poses):
    pass     

if __name__ == "__main__":
    if os.path.exists("out/scenes"):
        rmtree("out/scenes")
    if os.path.exists("temp"):
        rmtree("temp")
    os.makedirs("out/scenes", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    for scene in os.listdir(args.dataset):
        scene_path = os.path.join(args.dataset, scene)
        scene_index = scene.split("_")[-1]
        out_scene_path = os.path.join("out", "scenes", f"scene_{scene_index}.usd")
        viewer, world = main(out_scene_path, scene_index)
        reset_scene(args, viewer, world)
    
    for scene in os.listdir("out/scenes"):
        scene_index = scene.strip(".usd").split("_")[-1]
        scene_path = os.path.join("out", "scenes", f"scene_{scene_index}.usd")
        temp_path = os.path.join("temp", f"scene_{scene_index}")
        generate_data(args, temp_path, scene_path)
        while rep.orchestrator.get_is_started():
            simulation_app.update()
        while rep.orchestrator.get_is_started():
            simulation_app.update()
        for _ in range(100):
            simulation_app.update()


    

    


