# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from pathlib import Path
from isaaclab_assets import HUMANOID_CFG
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv
from isaaclab_tasks.direct.archerproject.archer_interactive_scene import ArcherSceneCfg
from .archer_waypoint import WAYPOINT_CFG

#pathing for assets
cwd = Path.cwd()

@configclass
class HumanoidEnvCfg(DirectRLEnvCfg):
    #env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0
    action_space = 21
    observation_space = 75
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt = 1 / 120, render_interval = decimation)

    # scene
    scene: ArcherSceneCfg = ArcherSceneCfg()

    # robot
    robot: ArticulationCfg = HUMANOID_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    joint_gears: list = [
        67.5000,  # lower_waist
        67.5000,  # lower_waist
        67.5000,  # right_upper_arm
        67.5000,  # right_upper_arm
        67.5000,  # left_upper_arm
        67.5000,  # left_upper_arm
        67.5000,  # pelvis
        45.0000,  # right_lower_arm
        45.0000,  # left_lower_arm
        45.0000,  # right_thigh: x
        135.0000,  # right_thigh: y
        45.0000,  # right_thigh: z
        45.0000,  # left_thigh: x
        135.0000,  # left_thigh: y
        45.0000,  # left_thigh: z
        90.0000,  # right_knee
        90.0000,  # left_knee
        22.5,  # right_foot
        22.5,  # right_foot
        22.5,  # left_foot
        22.5,  # left_foot
    ]
    
    heading_weight: float = 0.5
    up_weight: float = 0.1

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 2.0
    dof_vel_scale: float = 0.1

    death_cost: float = -1.0
    termination_height: float = 0

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01
    
class HumanoidEnv(LocomotionEnv):
    cfg: HumanoidEnvCfg

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot
