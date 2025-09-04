import argparse


def options():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--headless",
        type=bool,
        help="Running Program in headless mode",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "-sp",
        "--scene_paths",
        default="/share/scenes_3",
        type=str,
        help="Paths to scenes."
    )
    
    parser.add_argument(
        "-i",
        "--indices",
        default="",
        type=str,
        help="Index range of scenes."
    )

    parser.add_argument(
        "-r",
        "--robot_config",
        default="",
        type=str,
        help="Path to the robot config file."
    )

    arguments = parser.parse_args()
    return arguments

args = options()


# Launch Isaac Sim Simulator first before any other Omni imports
from omni.isaac.kit import SimulationApp

CONFIG = {"renderer": "RayTracedLighting", "headless": args.headless}
simulation_app = SimulationApp(CONFIG)

from omni.isaac.core import World

from viewer import SimViewer
from utils import get_scene_paths


def main():
    world = World()
    root_path = "/World/Workstation"

    scene_paths = get_scene_paths(args.scene_paths, args.indices)
    print(scene_paths)

    viewer = SimViewer(
        world=world, scene_paths=scene_paths, root_path=root_path, robot_config=args.robot_config
    )

    # Its recommended to always do a reset after adding your assets/prims
    # Initialize the physics simulation view and each added object to the Scene
    # All articulations should be added before the first reset is called
    # world.initialize_physics() und world.play() ist in reset() enthalten

    world.reset()
    viewer.reset()
    print(viewer.envs[0].robot.dof_properties)

    # Simulate for a few steps: this is a workaround to ensure that the textures are loaded.
    for _ in range(14):
        world.render()

    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if world.is_stopped() or viewer.finished():
            print("break")
            break

        # If simulation is paused, then skip.
        if not world.is_playing():
            # Without it, simulation would hang up.
            world.step(render=True)

            if viewer.next_scene():
                world.reset()
                viewer.reset()
                for _ in range(14):
                    world.render()

            continue

        # we have control over stepping physics and rendering in this workflow
        # execute one physics step and one rendering step
        world.step(render=True)

        if viewer.next_scene():
            world.reset()
            viewer.reset()
            for _ in range(14):
                world.render()

    if simulation_app.is_exiting():
        # close Isaac Sim
        simulation_app.close()


if __name__ == "__main__":
    main()
