import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="A NVIDIA Ominverse implementation to simulate random tabletops")

    parser.add_argument("--num_objects", default=15,  type=int, help="Number of generated objects")
    parser.add_argument("--num_views", default=10, type=int,
                         help="Number of generated synthetic views")
    parser.add_argument("--mode", choices=["random", "linear", "clustered", "custom"], type=str, help="Set generation for random objects")
    parser.add_argument("--custom_path", default="")
    parser.add_argument("--headless", action="store_true", help="run simulation in headless mode")
    parser.add_argument("--random_textures", action="store_true", help="adds random textures to simulated objects")
    parser.add_argument("--config", default="configs/objects.yaml")
    parser.add_argument("--out_dir", default="datasets")
    parser.add_argument("--save_scene", action="store_true")
    parser.add_argument("--num_scenes", default=5, help="", type=int)
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--container_index", required=True)

    return parser.parse_args()

args = parse_args()

from omni.isaac.kit import SimulationApp
CONFIG = {"renderer": "RayTracedLighting", "headless": args.headless}
simulation_app = SimulationApp(CONFIG)

import omni
import omni.isaac.core.utils.prims as prims_utils
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


def custom_main(object_usd_path: str, label: str):
    world = World()
    root_path = "/World/Workstation"
    flag = False

    scene = Scene.custom_scene(root_path=root_path, custom_obj=object_usd_path)
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

        if viewer.step_time > 0.2 and not flag:
            stage = get_current_stage()
            prim_path = f"/World/Workstation_0/{list(viewer.names)[0]}"
            prim = stage.GetPrimAtPath(prim_path)
            bbox = omni.usd.get_context().compute_path_world_bounding_box(prim_path)
            curr_xmin, curr_ymin, curr_xmax, curr_ymax = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
            curr_zmin, curr_zmax = bbox[0][2], bbox[1][2]
            height = (curr_zmax - curr_zmin) / 2

            new_x = (curr_xmin + curr_xmax) / 2
            new_x = new_x * (abs(new_x) > 0.01)
            new_y = (curr_ymin + curr_ymax) / 2
            new_y = new_y * (abs(new_y) > 0.01)
            new_z = (curr_zmin + curr_zmax) / 2

            new_z = viewer.plane_z + 0.5 - (new_z - viewer.plane_z) + height

            print(bbox)
            print(abs(new_x), abs(new_y), new_z)
            if new_x != 0.0 or new_y != 0.0:
                new_x, new_y = -new_x, -new_y
            translation = Gf.Vec3d(new_x, new_y, new_z)
            success = prim.GetAttribute("xformOp:translate").Set(translation, 0)
            print(success)
            flag = True
        if viewer.step_time > 0.5:
            break
    # Simulation exit and clean-up
    stage = get_current_stage()
    stage.Export(f"out/scenes/{label}.usd")

    return viewer, world


def main(scene_path, yaml_path):
    world = World()
    root_path = "/World/Workstation"
    print(get_assets_root_path())
    with open(args.config, "r") as f:
        data_dict = safe_load(f)

    num_objects = np.random.randint(5, 20)
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
    stage = get_current_stage()
    stage.Export(f"{scene_path}/{COUNT}.usd")
    if args.save_scene:
        viewer.save_scene(COUNT, yaml_path)
        print("======= exiting Simulation =========")
    return viewer, world

def clean_data_set(args):
    num_frames = len(os.listdir(os.path.join("temp", os.listdir("temp")[0]))) // 5
    numbers = []
    for _ in range(num_frames):
                number = "{}".format(_)
                while len(number) < 4:
                    number = "0" + number
                numbers.append(number)
    for scene in os.listdir("temp"):     
        scene_index = scene.split("_")[0]   
        path = os.path.join("temp", scene)
        for index, number in enumerate(numbers):
            os.makedirs(os.path.join("dataset", scene, f"frame_{index}"))

            cam_params = "camera_params_{}.json".format(number)
            depth =  "distance_to_camera_{}.npy".format(number)
            rgb = "rgb_{}.png".format(number)
            mask = "semantic_segmentation_{}.png".format(number)
            mask_label = "semantic_segmentation_labels_{}.json".format(number)


            move(os.path.join(path, cam_params), os.path.join("dataset", scene, f"frame_{index}"))
            #move(os.path.join(path, depth), os.path.join("dataset", scene, f"frame_{index}"))
            depth_image = np.load(os.path.join(path, depth))
            comp_depth = (65535*(depth_image - depth_image.min())/depth_image.ptp()).astype(np.uint16)
            imageio.imwrite(os.path.join("dataset", scene, f"frame_{index}", "depth.png"), comp_depth)
            move(os.path.join(path, rgb), os.path.join("dataset", scene, f"frame_{index}"))
            move(os.path.join(path, mask), os.path.join("dataset", scene, f"frame_{index}"))
            move(os.path.join(path, mask_label), os.path.join("dataset", scene, f"frame_{index}"))
        move(os.path.join("out", "scenes", "{}.usd".format(scene_index)), os.path.join("dataset", scene, "scene.usd"))
        if args.save_scene:
            move(os.path.join("out", "yaml", "{}.yaml".format(scene_index)), os.path.join("dataset", scene, "scene.yaml"))
            
            
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

def generate_custom(args, path):
    OUTDIR = path.strip(".usd")
    open_stage(os.path.join("out/scenes", path))
    stage = get_current_stage()
    exceptions = set(["table", "Plane", "Enviroment"])



    for i in range(100):
        if i % 10 == 0:
            print(f"updatimg SimulationApp: {i}")
        simulation_app.update()

    poses = []
    for _ in range(12):
        theta = 0 + _ *((2 * pi) / 12) 
        x = 0.7 * cos(theta)
        y = 0.7 * sin(theta)
        z = 1.2
        poses.append((x, y, z))

    camera = rep.create.camera(
        focus_distance=1,
        clipping_range = (0.001, 100000)
    )
    plane = rep.create.plane(position=(0,0,0.1), scale = 100)
    dome_light = rep.create.light(
        light_type="dome",
        temperature=6500,
        intensity=500,
        rotation=(0, 0, -90),
        position=(0, 0, 10)
    )
    prims: List[Usd.Prim] = [x for x in stage.Traverse() if x.IsA(UsdGeom.Mesh)]
    scene_objects = [str(prim.GetPath()) for prim in prims]
    scene_names = [element.split("/")[3] for element in scene_objects]
    prims = [(rep.get.prims(path_pattern=name), name) for name in scene_names if name not in exceptions]
    table = [rep.get.prims(path_pattern=name) for name in scene_names if "table" in name][0]

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir= f"{os.path.join(os.getcwd(), 'temp', OUTDIR)}",
        rgb=True
    )
    render_product = rep.create.render_product(camera, (512, 512))
    writer.attach([render_product])

    with rep.trigger.on_frame(max_execs=12, rt_subframes=10):
        with camera:
            rep.modify.pose(
                position=rep.distribution.sequence(poses),
                look_at=(0.0, 0.0, 0.95)
            )
        with table:
            rep.modify.visibility(False)
        for item in prims:
            prim, name = item
            with prim:
                rep.randomizer.texture(textures=[
                        os.path.join("/share/textures", item) for item in os.listdir("/share/textures")
                    ])
    
    rep.orchestrator.run()
    simulation_app.update()
    

def generate_data(args, scene_path, path, temp_path):
    print(f"====== generating synthetic dataset with {args.num_views} views =======")

    curr_date = datetime.today().strftime('%Y%m%d')
    scene_index = path.split(".")[0]
    OUTDIR = "{}_{}".format(scene_index, curr_date)

    open_stage(os.path.join(scene_path, path))
    stage = get_current_stage()

    for i in range(150):
        if i % 10 == 0:
            print(f"updating SimulationApp: {i}")
        simulation_app.update()
    focus_distance = 4

    camera = rep.create.camera(
        position=(2, 2, 2),
        look_at=(0, 0, 0),
        focus_distance=focus_distance
    )
    dome_light = rep.create.light(
        light_type="dome",
        temperature=6500,
        intensity=500,
        rotation=(0, 0, -90),
        position=(0, 0, 10)
    )

    prims: List[Usd.Prim] = [x for x in stage.Traverse() if x.IsA(UsdGeom.Mesh)]
    scene_objects = [str(prim.GetPath()) for prim in prims]
    scene_names = [element.split("/")[3] for element in scene_objects]
    prims = [(rep.get.prims(path_pattern=name), name) for name in scene_names if "table" not in name and "Enviroment" not in name]
    table = [rep.get.prims(path_pattern=name) for name in scene_names if "table" in name][0]

    bounding_box = omni.usd.get_context().compute_path_world_bounding_box("/World/Workstation_0/table")


    
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=f"/omniverse/{temp_path}/{OUTDIR}",
        rgb=True,
        semantic_segmentation=True,
        distance_to_camera=True,
        camera_params=True
    )
    render_product = rep.create.render_product(camera, (1920, 1080))
    writer.attach([render_product])
    
    num_views = np.random.randint(20, 60)
    with rep.trigger.on_frame(max_execs=num_views, rt_subframes=15):
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
                rep.modify.semantics([('class', f'{name}')])
                rep.randomizer.texture(textures=[
                        os.path.join("/share/textures", item) for item in os.listdir("/share/textures")
                    ])
    
    rep.orchestrator.run()
    simulation_app.update()        

if __name__ == "__main__":
    container_index = args.container_index
    scene_path = f"out/scenes/{container_index}_scenes"
    yaml_path = f"out/yaml/{container_index}_yaml"
    temp_path = f"temp/{container_index}_temp"
    if os.path.exists(scene_path):
        rmtree(scene_path)
    if os.path.exists(yaml_path):
        rmtree(yaml_path)
    if os.path.exists(temp_path):
        rmtree(temp_path)
    os.makedirs(scene_path, exist_ok=True)
    os.makedirs(yaml_path, exist_ok=True)
    os.makedirs(temp_path, exist_ok=True)
    if args.mode == "custom":
        try:
            model = args.model
            path = f"ModelNet40/{model}_converted"
            for folder in os.listdir(os.path.join(path,)):
                for obj in os.listdir(os.path.join(path, folder)):
                    if ".usd" in obj:
                        viewer, world = custom_main(os.path.join(path, folder, obj), f"{folder}_{obj.strip('.usd')}")
                        reset_scene(args, viewer, world)
            for path in os.listdir("out/scenes"):
                rep_count = len(os.listdir("temp"))
                scene_count = len(os.listdir("out/scenes"))
                print(f"processing scene {path}: {rep_count} out of {scene_count} ")
                generate_custom(args, path)
                while rep.orchestrator.get_is_started():
                        simulation_app.update()
            os.makedirs(f"dataset/{model}/test")
            os.makedirs(f"dataset/{model}/train")
            for folder in os.listdir("temp"):
                if "test" in folder:
                    move(os.path.join("temp", folder), f"dataset/{model}/test")
                else:
                    move(os.path.join("temp", folder), f"dataset/{model}/train")
        except Exception as e:
            print(e)

    
    else:
        try:
            for _ in range(args.num_scenes):
                viewer, world = main(scene_path, yaml_path)
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
            pass
            print(e)
