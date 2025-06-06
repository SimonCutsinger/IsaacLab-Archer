
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
#Assets
cwd = Path.cwd()
@configclass
class ArcherSceneCfg(InteractiveSceneCfg):

    num_envs = 1024
    env_spacing = 30
    replicate_physics = True

    maze = AssetBaseCfg(
        prim_path = "{ENV_REGEX_NS}/Maze",
        spawn = sim_utils.UsdFileCfg(usd_path = f"{cwd}\\source\\isaaclab_tasks\\isaaclab_tasks\\direct\\archerproject\\archer_assets\\maze_with_chest_2.usd"),
        init_state = AssetBaseCfg.InitialStateCfg(pos = (0, 0, 0), rot = (0, 0, 0, 0)),
        collision_group = 0
    )

    light = AssetBaseCfg(
        prim_path = "/World/light",
        spawn = sim_utils.DomeLightCfg(color = (0.75, 0.75, 0.75), intensity = 2500.0),
)
