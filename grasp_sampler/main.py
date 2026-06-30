"""Main script for parallel grasp sampling across multiple scenes."""

import argparse
import contextlib
import logging
import multiprocessing as mp
import os
import signal
import traceback
from typing import List, Tuple, Union

from tqdm import tqdm

from sampler import GraspSampler
from scene import Scene
from utlis import get_scene_paths, load_yaml, print_options

logger = logging.getLogger(__name__)

# Constants
SKIP_TABLE_NAME = "Dellwood_DiningTable"
LOG_DIR = "logs"

# Type alias for worker results
StatusResult = Union[
    Tuple[str, str, str],        # ("done"/"skipped", path, log_path)
    Tuple[str, str, str, str],   # ("error", path, log_path, error_msg)
]


def hard_exit(sig, frame) -> None:
    """Kills the entire process group upon receiving SIGINT."""
    print("\nSIGINT received, killing entire process group ...", flush=True)
    os.killpg(os.getpgrp(), signal.SIGKILL)


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        Namespace containing all configuration parameters.
    """
    parser = argparse.ArgumentParser(
        description="Grasp sampling for scenes with parallel processing."
    )

    parser.add_argument(
        "-np", "--num_points",
        type=int,
        default=10_000,
        help="Number of points to sample on the mesh.",
    )

    parser.add_argument(
        "-mg", "--max_grasps",
        type=int,
        default=2500,
        help="Number of maximal grasps to sample.",
    )

    parser.add_argument(
        "-mr", "--max_rotations",
        type=int,
        default=4,
        help="Number of maximal rotations per grasp.",
    )

    parser.add_argument(
        "-f", "--friction",
        type=float,
        default=0.4,
        help="Set the friction coefficient.",
    )

    parser.add_argument(
        "-vs", "--visualize_single",
        action="store_true",
        help=(
            "Visualize a single result. "
            "Recommended only for single-scene processing."
        ),
    )

    parser.add_argument(
        "-va", "--visualize_all",
        action="store_true",
        help=(
            "Visualize all results. "
            "Recommended only for single-scene processing."
        ),
    )

    parser.add_argument(
        "-s", "--scenes_path_yml",
        default="scenes/",
        type=str,
        help="Path to scene yml configuration directory.",
    )

    parser.add_argument(
        "-i", "--indices",
        default="",
        type=str,
        help="Index range of scenes, e.g. '0-5' for a range or '1,3,5' for specific indices.",
    )

    parser.add_argument(
        "-nw", "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes (default: number of CPU cores).",
    )

    return parser.parse_args()


def process_scene(args_and_path: Tuple[argparse.Namespace, str]) -> StatusResult:
    """
    Processes a single scene: loads it, samples grasps, and saves results.

    Stdout/stderr are redirected to a log file.

    Args:
        args_and_path: Tuple of (configuration arguments, path to scene YAML).

    Returns:
        Status tuple containing result status, path, and log path.
    """
    args, path = args_and_path

    scene_dir = os.path.dirname(path)
    scene_name = os.path.basename(scene_dir)
    log_path = os.path.join(LOG_DIR, f"{scene_name}.log")

    os.makedirs(LOG_DIR, exist_ok=True)

    with open(log_path, "w", buffering=1) as log_file, \
         contextlib.redirect_stdout(log_file), \
         contextlib.redirect_stderr(log_file):

        scene = None
        sampler = None

        try:
            config = load_yaml(path)

            if config["table"]["name"] == SKIP_TABLE_NAME:
                print(f"Skipping due to {SKIP_TABLE_NAME}: {path}")
                return ("skipped", path, log_path)

            print(f"Starting sampling for {path}")
            scene = Scene.from_dict(config)
            scene.show()

            sampler = GraspSampler(args, scene, scene_path_yml=path)
            sampler.sample()

            print(f"Successfully completed for {path}")
            return ("done", path, log_path)

        except Exception as e:
            print(f"ERROR at {path}: {repr(e)}")
            traceback.print_exc()
            return ("error", path, log_path, repr(e))

        finally:
            del scene
            del sampler


def run_parallel(args: argparse.Namespace, scene_paths: List[str]) -> None:
    """
    Runs grasp sampling in parallel for all scenes.

    Args:
        args: Configuration arguments.
        scene_paths: List of paths to scene YAML files.
    """
    n_scenes = len(scene_paths)
    num_workers = min(args.num_workers, n_scenes)
    print(f"Starting with {num_workers} processes for {n_scenes} scenes.")

    work_items = [(args, path) for path in scene_paths]

    with mp.Pool(processes=num_workers) as pool:
        results = pool.imap_unordered(process_scene, work_items, chunksize=1)

        for res in tqdm(results, total=n_scenes, desc="Sampling scenes"):
            _log_result(res)


def _log_result(res: StatusResult) -> None:
    """
    Prints the result of a processed scene to the console.

    Args:
        res: Status tuple from the worker.
    """
    status = res[0]
    path = res[1]
    log_path = res[2]
    scene_name = os.path.basename(os.path.dirname(path))

    if status == "done":
        print(f"[OK]    {scene_name} ({path}) -> Log: {log_path}")
    elif status == "skipped":
        print(f"[SKIP]  {scene_name} ({path}) -> Log: {log_path}")
    elif status == "error":
        err = res[3]
        print(f"[ERROR] {scene_name} ({path}) -> Log: {log_path}")
        print(f"        Error: {err}")


if __name__ == "__main__":
    # Create own process group for this script and all child processes
    os.setpgrp()
    signal.signal(signal.SIGINT, hard_exit)

    args = parse_arguments()
    print_options(args)

    scene_paths = get_scene_paths(args.scenes_path_yml, args.indices)
    n_scenes = len(scene_paths)
    print(f"{n_scenes} scenes found.")

    if not scene_paths:
        raise RuntimeError("No scenes found.")

    run_parallel(args, scene_paths)