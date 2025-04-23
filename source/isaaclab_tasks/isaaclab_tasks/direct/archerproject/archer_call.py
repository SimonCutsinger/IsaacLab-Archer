"""
Notes:
- Make a scene
- Add in the humanoid robot
- Use skrl play
- Export the training data (for skrl it's the pt for play as there is no export after launching it)
    - If I can choose a checkpoint based on time made I can make checkpoint learning automated but for now it's manual
- Maybe use issac-sim with policy instead of a isaaclab scene made from scratch
    - Found H1 example policy
    - Policies are a lot more complicated than I thought
        - You need to define every aspect used in training in the 'policy loader' (not sure what to call it)
        - I need to access the policy on their webserver to fully understand what's going on
- Scene from scratch ends up becoming a training scene with only the scene being changed
    - Should test anyways
    - Make sure prim path is correct for chest (important)
"""

#code starts

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


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

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