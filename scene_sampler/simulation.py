import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="A NVIDIA Ominverse implementation to simulate random tabletops")

    parser.add_argument("--num_objects", default=15,  type=int, help="Number of generated objects")
    parser.add_argument("--num_views", default=10, type=int,
                         help="Number of generated synthetic views")
    parser.add_argument("--mode", choices=["random", "linear", "clustered", "add_images", "replace_images"], type=str, help="Set generation for random objects")
    parser.add_argument("--custom_path", default="")
    parser.add_argument("--headless", action="store_true", help="run simulation in headless mode")
    parser.add_argument("--random_textures", action="store_true", help="adds random textures to simulated objects")
    parser.add_argument("--config", default="configs/objects.yaml")
    parser.add_argument("--out_dir", default="datasets")
    parser.add_argument("--save_scene", action="store_true")
    parser.add_argument("--num_scenes", default=5, help="", type=int, required=False)
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--container_index", required=True)

    return parser.parse_args()

args = parse_args()

from omni.isaac.kit import SimulationApp
CONFIG = {"renderer": "RayTracedLighting", "headless": args.headless}
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
from typing import List, Tuple
from PIL import Image
import os
import imageio
import numpy as np
from shutil import move, rmtree
from datetime import datetime
import omni.replicator.core as rep


COUNT = 0

def run_orchestrator():
    rep.orchestrator.run()

    while not rep.orchestrator.get_is_started():
        simulation_app.update()

    while rep.orchestrator.get_is_started():
        simulation_app.update()

    rep.BackendDispatch.wait_until_done()
    rep.orchestrator.stop()



def generate_scenes(scene_path, yaml_path):
    world = World()
    root_path = "/World/Workstation"
    print(get_assets_root_path())
    with open(args.config, "r") as f:
        data_dict = safe_load(f)

    num_objects = np.random.randint(10, 20)
    # TODO: Create random object loader
    scene = Scene.from_dict(data_dict, root_path, num_objects=num_objects)
    viewer = Viewer(
        world=world, scene=scene, root_path=root_path, mode=args.mode
    )
    world.reset()
    viewer.post_reset()
    for _ in range(14):
        world.render()
    while simulation_app.is_running():
        viewer.step_time += 1.0 / 60.0
        if world.is_stopped():
            print("break")
            break

        if not world.is_playing():
            world.step(render=True)
            continue

        world.step(render=True)

        if viewer.step_time > 20.0:
            break
    # Simulation exit and clean-up
    stage = omni.usd.get_context().get_stage()
    stage.Export(f"{scene_path}/{COUNT}.usd")
    if args.save_scene:
        viewer.save_scene(COUNT, yaml_path)
        print("======= exiting Simulation =========")
    return viewer, world

def mod_scenes(scene_path, scene_index):
    world = World()
    root_path = "/World/Workstation"

    print(get_assets_root_path())

    with open(os.path.join("/dataset", f"scene_{scene_index}", "scene.yaml")) as f:
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

    stage = omni.usd.get_context().get_stage()
    stage.Export(scene_path)
    print("======= exiting Simulation =========")
    return viewer, world
            
def add_plane(
        stage: Usd.Stage,
        path: str,
        size: Tuple[float, float],
        uv: Tuple[float, float]=(1, 1)

):
    stage.DefinePrim(path, "Xform")

    prim = UsdGeom.Mesh.Define(stage, os.path.join(path, "mesh"))
    prim.CreateExtentAttr().Set([
        (-size[0], -size[1], 0.2),
        (size[0], size[0], 0.2)
    ])
    prim.CreateFaceVertexCountsAttr().Set([4])
    prim.CreateFaceVertexIndicesAttr().Set([0, 1, 2, 3])

    var = UsdGeom.Primvar(prim.CreateNormalsAttr())
    var.Set([(0, 0, 1)] * 4)
    var.SetInterpolation(UsdGeom.Tokens.faceVarying)

    var = UsdGeom.PrimvarsAPI(prim).CreatePrimvar("primvars:st",
        Sdf.ValueTypeNames.Float2Array)

    var.Set(
        [(0, 0), (uv[0], 0), (uv[0], uv[1]), (0, uv[1])]
    )

    prim.CreatePointsAttr().Set([
        (-size[0], size[1], 0),
        (size[0], -size[1], 0),
        (-size[0], size[1], 0),
        (size[0], size[1], 0)
    ])

    prim.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

    return stage.GetPrimAtPath(path)

def reset_scene(args, viewer, world):
    for _object in viewer.objects:
        prims_utils.delete_prim(_object.prim_path)
    prims_utils.delete_prim("/World/Workstation_0/table")
    prims_utils.delete_prim("/World/Workstation_0")
    prims_utils.delete_prim("/World")
    world.reset()

    for _ in range(100):
        world.render()
    

def generate_data(args, scene_path, path, temp_path):
    print(f"====== generating synthetic dataset with {args.num_views} views =======")

    curr_date = datetime.today().strftime('%Y%m%d')
    scene_index = path.split(".")[0]
    OUTDIR = "{}_{}".format(scene_index, curr_date)

    settings = carb.settings.get_settings()
    settings.set("rtx/ambientOcclusion/enabled", True)

    open_stage(os.path.join(scene_path, path))
    stage = omni.usd.get_context().get_stage()

    for i in range(100):
        if i % 10 == 0:
            print(f"updating SimulationApp: {i}")
        simulation_app.update()
    focus_distance = 4

    camera = rep.create.camera(
        position=(2, 2, 2),
        look_at=(0, 0, 0),
        focus_distance=focus_distance,
        clipping_range = (0.01, 1000)
    )
    ground_plane = rep.create.plane(
        position=(0, 0, 0.1),
        scale=100,
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
    _table = [prim for prim in prims if "table" in str(prim.GetPath())][0]
    prims = [(rep.get.prims(path_pattern=name), name) for name in scene_names if "table" not in name and 'Enviroment' not in name and "Plane" not in name]
    table = [rep.get.prims(path_pattern=name) for name in scene_names if "table" in name][0]

    aabb = omni.usd.get_context().compute_path_world_bounding_box(str(_table.GetPath()))

    scale = np.array(aabb[1]) - np.array(aabb[0])
    x, y, z = scale
    print(x, y, z)

    print(aabb)

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=f"/omniverse/{temp_path}/{OUTDIR}",
        rgb=True,
        semantic_segmentation=True,
        distance_to_image_plane=True,
        camera_params=True
    )
    render_product = rep.create.render_product(camera, (1920, 1080))
    writer.attach([render_product])
    
    num_views = args.num_views
    textures = [os.path.join("/share/textures", item) for item in os.listdir("/share/textures")]

    with rep.trigger.on_frame(max_execs=num_views, rt_subframes=80):
        with table:
            rep.modify.semantics([('class', 'table')])
            rep.randomizer.materials(
                    rep.create.material_omnipbr(
                        diffuse_texture=rep.distribution.choice(textures),
                        project_uvw=True,
                        roughness=rep.distribution.uniform(0.2, 0.8),
                        metallic=rep.distribution.uniform(0.0, 0.2)
                    )
                )
        with camera:
            rep.modify.pose(
                position=rep.distribution.uniform((-2*x, -2*y, 1.5), (2*x, 2*y, 3)),
                look_at=rep.distribution.uniform((0, 0, z), (0, 0, z))
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

def add_images(args, temp_path, scene_path):
    print(f"====== generating synthetic dataset with {args.num_views} views =======")

    settings = carb.settings.get_settings()
    settings.set("rtx/ambientOcclusion/enabled", True)

    open_stage(scene_path)
    stage = omni.usd.get_context().get_stage()

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
        scale=100,
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

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=f"/omniverse/{temp_path}",
        rgb=True,
        semantic_segmentation=True,
        distance_to_image_plane=True,
        camera_params=True
    )
    render_product = rep.create.render_product(camera, (1920, 1080))
    writer.attach([render_product])
        
    num_views = args.num_views
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

if __name__ == "__main__":
    if args.mode == "add_images" or args.mode == "replace_images":
        if os.path.exists("out/scenes"):
            rmtree("out/scenes")
        if os.path.exists("temp"):
            rmtree("temp")
        os.makedirs("out/scenes", exist_ok=True)
        os.makedirs("temp", exist_ok=True)
        try:
            for scene in os.listdir("/dataset"):
                scene_path = os.path.join("/dataset", scene)
                print(scene_path)
                scene_index = scene.split("_")[-1]
                out_scene_path = os.path.join("out", "scenes", f"scene_{scene_index}.usd")
                viewer, world = mod_scenes(out_scene_path, scene_index)
                reset_scene(args, viewer, world)
            
            for scene in os.listdir("out/scenes"):
                scene_index = scene.strip(".usd").split("_")[-1]
                scene_path = os.path.join("out", "scenes", f"scene_{scene_index}.usd")
                temp_path = os.path.join("temp", f"scene_{scene_index}")
                add_images(args, temp_path, scene_path)
                while rep.orchestrator.get_is_started():
                    simulation_app.update()
                while rep.orchestrator.get_is_started():
                    simulation_app.update()
                for _ in range(100):
                    simulation_app.update()
        except Exception as e:
            print(e)
    else:
        if os.path.exists("temp"):
            rmtree("temp")
        if os.path.exists("out/scenes"):
            rmtree("out/scenes")
        if os.path.exists("out/yaml"):
            rmtree("out/yaml")
        container_index = args.container_index
        scene_path = f"out/scenes/{container_index}_scenes"
        yaml_path = f"out/yaml/{container_index}_yaml"
        temp_path = f"temp/{container_index}_temp"
        data_path = f"/dataset"
        os.makedirs(scene_path, exist_ok=True)
        os.makedirs(yaml_path, exist_ok=True)
        os.makedirs(temp_path, exist_ok=True)
        os.makedirs(data_path, exist_ok=True)
        try:
            for _ in range(args.num_scenes):
                viewer, world = generate_scenes(scene_path, yaml_path)
                COUNT += 1
                reset_scene(args, viewer, world)
            for path in os.listdir(scene_path):
                generate_data(args, scene_path, path, temp_path)
                while rep.orchestrator.get_is_started():
                    simulation_app.update()
            while rep.orchestrator.get_is_started():
                simulation_app.update()
            for _ in range(100):
                simulation_app.update()
        except Exception as e:
            print(e)
