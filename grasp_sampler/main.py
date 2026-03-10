from sampler import GraspSampler
from scene import Scene
from utlis import print_options, load_yaml, get_scene_paths

import argparse
import multiprocessing as mp
from tqdm import tqdm
import os
import contextlib
import signal
import traceback


def hard_exit(sig, frame):
    print("\nSIGINT erhalten, kill komplette Prozessgruppe ...", flush=True)
    os.killpg(os.getpgrp(), signal.SIGKILL)

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
    
    parser.add_argument(
        "--visualize_single",
        action="store_true",
        help=(
            "Set to true if we want to visualize a single result. "
            "Recommended only when processing a single scene; "
            "using this with multiple processes may open many windows."
        ),
    )

    parser.add_argument(
        "-va",
        "--visualize_all",
        action="store_true",
        help=(
            "Set to true if we want to visualize all results. "
            "Recommended only when processing a single scene; "
            "using this with multiple processes may open many windows."
        ),
    )
    
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


    parser.add_argument(
        "-nw",
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes (default: number of CPU cores)",
    )
    
    arguments = parser.parse_args()
    
    return arguments


def process_scene(args_and_path):
    """
    """
    args, path = args_and_path

    scene_dir = os.path.dirname(path)
    scene_name = os.path.basename(scene_dir)
    log_path = os.path.join("logs", f"{scene_name}.log")

    with open(log_path, "w", buffering=1) as log_file, \
         contextlib.redirect_stdout(log_file), \
         contextlib.redirect_stderr(log_file):

        try:
            dic = load_yaml(path)
            if dic["table"]["name"] == "Dellwood_DiningTable":
                print(f"Szenen-Log: Überspringe wegen Dellwood_DiningTable: {path}")
                return ("skipped", path, log_path)

            print(f"Szenen-Log: Starte Sampling für {path}")
            scene = Scene.from_dict(dic)
            scene.show()

            sampler = GraspSampler(args, scene, scene_path_yml=path)
            sampler.sample()

            print(f"Szenen-Log: Erfolgreich fertig für {path}")
            status = ("done", path, log_path)

            '''
            except Exception as e:
                # Stacktrace etc. geht in die Logdatei
                print(f"Szenen-Log: FEHLER bei {path}: {repr(e)}")
                status = ("error", path, log_path, repr(e))
            '''
        
        except Exception as e:
            print(f"Szenen-Log: FEHLER bei {path}: {repr(e)}")
            # Stacktrace (inklusive Datei/Zeile) ausgeben:
            traceback.print_exc()
            status = ("error", path, log_path, repr(e))
            
        finally:
            try:
                del scene
            except NameError:
                pass
            try:
                del sampler
            except NameError:
                pass

    return status


if __name__ == '__main__':
    # eigene Prozessgruppe für dieses Script + alle seine Kinder
    os.setpgrp()
    signal.signal(signal.SIGINT, hard_exit)

    args = options()
    print_options(args)

    scene_paths = get_scene_paths(args.scenes_path_yml, args.indices)
    n_scenes = len(scene_paths)
    print(f"{n_scenes} Scenes")

    if not scene_paths:
        raise RuntimeError("Keine Szenen gefunden.")

    num_workers = args.num_workers
    num_workers = min(num_workers, n_scenes)
    print(f"Starte mit {num_workers} Prozessen für {n_scenes} Szenen.")

    work_items = [(args, path) for path in scene_paths]

    pool = mp.Pool(processes=num_workers)

    for res in tqdm(
        pool.imap_unordered(process_scene, work_items, chunksize=1),
        total=len(work_items),
        desc="Sampling scenes"
    ):
        status = res[0]
        path = res[1]
        log_path = res[2]
        scene_dir = os.path.dirname(path)
        scene_name = os.path.basename(scene_dir)

        if status == "done":
            print(f"[OK]    {scene_name} ({path}) -> Log: {log_path}")
        elif status == "skipped":
            print(f"[SKIP]  {scene_name} ({path}) -> Log: {log_path}")
        elif status == "error":
            err = res[3]
            print(f"[ERROR] {scene_name} ({path}) -> Log: {log_path}")
            print(f"        Fehler: {err}")


'''
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
        if args.show:
            scene.show()
        sampler = GraspSampler(args, scene, scene_path_yml=path)
        sampler.sample()
        del scene
'''
