# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to create a simple stage in Isaac Sim.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from isaaclab.sim import SimulationCfg, SimulationContext

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils

from isaaclab_assets import CARTPOLE_CFG
from isaaclab.assets import Articulation

def design_scene():

    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""

    # Ground-plane

    cfg_ground = sim_utils.GroundPlaneCfg()

    #cfg_ground.func("/World/defaultGroundPlane", cfg_ground)


    # spawn distant light

    cfg_light = sim_utils.DomeLightCfg(

        intensity=3000.0,

        color=(0.75, 0.75, 0.75),

    )

    cfg_light.func("/World/Light", cfg_light)

    #creating 2 orgins for each robot
    origins = [[0.0, 0.0, -1.0], [-1.0, 0.0, -1.0]]
    #origin 1
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    #origin 2
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])
    #other objects prim creation
    prim_utils.create_prim("/World/Objects", "Xform")

    # Articulation
    cartpole_cfg = CARTPOLE_CFG.copy()
    cartpole_cfg.prim_path = "/World/Origin.*/Robot"
    cartpole = Articulation(cfg=cartpole_cfg)

    cfg = sim_utils.UsdFileCfg(usd_path=f"C:/Users/start/Documents/archer/IsaacLab/archerproject/archer_assets/test_blocks.usd")
    cfg.func("/World/Objects/Blocks", cfg, translation=(-5.0, -5.0, 0.0))
    scene_entities = {"cartpole": cartpole}

    return scene_entities, origins

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    robot = entities["cartpole"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            robot.reset()
            print("[INFO]: Resetting robot state...")
        # Apply random action
        # -- generate random joint efforts
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # -- apply action to the robot
        robot.set_joint_effort_target(efforts)
        # -- write data to sim
        robot.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        robot.update(sim_dt)
def main():
    """Main function."""
    #Loading kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene by adding assets to it
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    #new simulator while loop
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
