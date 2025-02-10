
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg

#Assets
cwd = Path.cwd()
@configclass
class ArcherSceneCfg(InteractiveSceneCfg):
    # terrain - maze
    num_envs = 4096
    env_spacing = 4
    replicate_physics = False
    terrain = TerrainImporterCfg(
        
        #prim_path="/World/envs/env_.*/Ground",
        prim_path="/World/ground",
        terrain_type= "usd",
        usd_path = f"{cwd}\\source\\isaaclab_tasks\\isaaclab_tasks\\direct\\archerproject\\archer_assets\\test_blocks.usd",
        collision_group=-1,
        
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
    )