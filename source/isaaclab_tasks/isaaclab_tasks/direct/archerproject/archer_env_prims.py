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

"""Rest everything follows."""

from isaaclab.sim import SimulationCfg, SimulationContext

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

def design_scene():

    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""

    # Ground-plane

    cfg_ground = sim_utils.GroundPlaneCfg()

    #cfg_ground.func("/World/defaultGroundPlane", cfg_ground)


    # spawn distant light

    cfg_light_distant = sim_utils.DistantLightCfg(

        intensity=3000.0,

        color=(0.75, 0.75, 0.75),

    )

    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))


    # create a new xform prim for all objects to be spawned under

    prim_utils.create_prim("/World/Objects", "Xform")

    # spawn a red cone

    cfg_cone = sim_utils.ConeCfg(

        radius=0.15,

        height=0.5,

        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),

    )

    cfg_cone.func("/World/Objects/Cone1", cfg_cone, translation=(-1.0, 1.0, 1.0))

    cfg_cone.func("/World/Objects/Cone2", cfg_cone, translation=(-1.0, -1.0, 1.0))


    # spawn a green cone with colliders and rigid body

    cfg_cone_rigid = sim_utils.ConeCfg(

        radius=0.15,

        height=0.5,

        rigid_props=sim_utils.RigidBodyPropertiesCfg(),

        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),

        collision_props=sim_utils.CollisionPropertiesCfg(),

        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),

    )

    cfg_cone_rigid.func(

        "/World/Objects/ConeRigid", cfg_cone_rigid, translation=(-0.2, 0.0, 2.0), orientation=(0.5, 0.0, 0.5, 0.0)

    )

    # spawn a usd file of a table into the scene

    #cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    cfg = sim_utils.UsdFileCfg(usd_path=f"C:/Users/start/Documents/archer/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/archerproject/archer_assets/maze.usd")
    cfg.func("/World/Objects/Blocks", cfg, translation=(0.0, 0.0, -3.0))

def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene by adding assets to it
    design_scene()
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
