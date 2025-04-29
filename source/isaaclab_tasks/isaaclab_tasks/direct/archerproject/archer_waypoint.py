import isaaclab.sim as sim_utils
from dataclasses import MISSING
from isaaclab.assets import ArticulationCfg

##
# configuration
##

WAYPOINT_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Maze/MainScene/Geometry/Grid/Tiles/Chest_Clone",
    #spawn = sim_utils.UsdFileCfg(usd_path="path")
    actuators=None
)