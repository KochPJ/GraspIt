from sampler import GraspSampler
from scene import Scene
from utlis import print_options, load_yaml, get_scene_paths

import argparse


def options():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-np",
                        "--num_points",
                        type=int,
                        default=10_000,
                        help="Number of points to sample on the mesh.")
    
    parser.add_argument("-mg",
                        "--max_grasps",
                        type=int,
                        default=2500,
                        help="Number of maximal grasps to sample.")
    
    parser.add_argument("-mr",
                        "--max_rotations",
                        type=int,
                        default=4,
                        help="Number of maximal rotations per grasps.")
    
    parser.add_argument("-f",
                        "--friction",
                        type=float,
                        default=0.4,
                        help="Set the friction coefficient.")
    
    parser.add_argument("-vs",
                        "--visualize_single",
                        action="store_true",
                        help="Set to true if we want to visualize a single result.")
    
    parser.add_argument("-va",
                        "--visualize_all",
                        action="store_true",
                        help="Set to true if we want to visualize all results.")
    
    parser.add_argument("-s",
                        "--scenes_path_yml",
                        default="scenes/",
                        type=str,
                        help="Path to scene yml conf.")
    
    parser.add_argument(
                        "-i",
                        "--indices",
                        default="",
                        type=str,
                        help="Index range of scenes, e.g. '0-5' for a range or '1,3,5' for specific indices."
    )

    arguments = parser.parse_args()
    
    return arguments


if __name__ == '__main__':
    args = options()
    print_options(args)

    scene_paths = get_scene_paths(args.scenes_path_yml, args.indices)
    print(f"{len(scene_paths)} Scenes")

    for i, path in enumerate(scene_paths):
        print("Path: ", path)
        print(f"Scene: {len(scene_paths)}/{i+1}")
#
        dic = load_yaml(path)
        if dic["table"]["name"] == "Dellwood_DiningTable":
            print(path)
            continue
         
        scene = Scene.from_dict(dic)
        #scene.show()
        sampler = GraspSampler(args, scene, scene_path_yml=path)
        sampler.sample()
        del scene
