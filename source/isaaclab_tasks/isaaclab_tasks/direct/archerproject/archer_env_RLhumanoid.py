'''
Issues with potential solutions:
- Robot runs past maze
    - Not sure if if needs less reward, a new reward to punish increased distance, harsher punishment for dying or rewards for entering the maze
    - Could also apply flynn's dot reward function with manual dots instead of using ambigious chest (could also be a chest reward issue)
- Robot isn't walking, it's skipping
    - Add sensors onto it's feet and reward it for leaving and touching the ground at the same time it's other feet do
    - Change movement rewards to have one or more additional foot/thigh/knee rewards based on their location (use heading_weight or up_weight as reference)
'''
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from pathlib import Path
from isaaclab_assets import HUMANOID_CFG
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv
from isaaclab_tasks.direct.archerproject.archer_interactive_scene import ArcherSceneCfg
import torch
from pxr import Usd, UsdGeom
#for pxr to work you need to run 'pip install usd-core' in your venv
#pathing for assets
cwd = Path.cwd()

@configclass
class HumanoidEnvCfg(DirectRLEnvCfg):
    #env
    target_prim_path: str = "/World/envs/env_.*/Maze/MainScene/Geometry/Grid/Tiles/Chest_Clone"  # Target prim path pattern
    episode_length_s = 60.0
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

    death_cost: float = -10.0
    termination_height: float = 0.85

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01
    proximity_reward_scale: float = 50.0  # Scaling factor for distance reward
    
class HumanoidEnv(LocomotionEnv):
    cfg: HumanoidEnvCfg

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot
        # Find chest_clone prim directly instead of using Articulation
        stage = self.sim.stage
        self.env_roots = []
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if "env_" in prim_path and prim_path.count("/") == 3:  # Ensure it's a direct child of /World/envs
                self.env_roots.append(prim_path)
        self.target_positions = self._find_target_positions(stage)
    def _find_target_positions(self, stage):
        target_positions = {}
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if "env_" in prim_path:
                target_path = f"{prim_path}/Maze/MainScene/Geometry/Grid/Tiles/Chest_Clone"
                chest_prim = stage.GetPrimAtPath(target_path)
                if chest_prim.IsValid():
                    xformable = UsdGeom.Xformable(chest_prim)
                    transform_matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    translation = transform_matrix.ExtractTranslation()
                    target_positions[prim_path] = translation
        return target_positions
    def _get_proximity_reward(self) -> torch.Tensor:        
        # proximity rewards needs to be here as it references the scene (target_path) for the chest
        # Calculate proximity reward
        humanoid_pos = self.robot.data.root_pos_w[:, :3]  # (x, y, z) position
        # target pos for each environment
        target_pos = torch.zeros_like(humanoid_pos)
        for i, env_path in enumerate(self.env_roots):
            target_pos[i] = torch.tensor(self.target_positions[env_path], device=self.device)
        # Euclidean distance calculation
        distance = torch.norm(humanoid_pos[:, :2] - target_pos[:, :2], p=2, dim=-1)
        proximity_reward = self.cfg.proximity_reward_scale * (1 - torch.tanh(distance))
        return proximity_reward
    def _get_rewards(self) -> torch.Tensor:
        # get rewards needs to call proximity reward to apply it to equation in compute rewards
        proximity_reward = self._get_proximity_reward()
        total_reward = compute_rewards(
            self.actions,
            self.reset_terminated,
            self.cfg.up_weight,
            self.cfg.heading_weight,
            self.heading_proj,
            self.up_proj,
            self.dof_vel,
            self.dof_pos_scaled,
            self.potentials,
            self.prev_potentials,
            self.cfg.actions_cost_scale,
            self.cfg.energy_cost_scale,
            self.cfg.dof_vel_scale,
            self.cfg.death_cost,
            self.cfg.alive_reward_scale,
            self.motor_effort_ratio,
            proximity_reward
        )
        return total_reward
# has to be outside class because base values are not defined in humanoidEnv class
def compute_rewards(
    actions: torch.Tensor,
    reset_terminated: torch.Tensor,
    up_weight: float,
    heading_weight: float,
    heading_proj: torch.Tensor,
    up_proj: torch.Tensor,
    dof_vel: torch.Tensor,
    dof_pos_scaled: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    actions_cost_scale: float,
    energy_cost_scale: float,
    dof_vel_scale: float,
    death_cost: float,
    alive_reward_scale: float,
    motor_effort_ratio: torch.Tensor,
    proximity_reward: torch.Tensor
):
    heading_weight_tensor = torch.ones_like(heading_proj) * heading_weight
    heading_reward = torch.where(heading_proj > 0.8, heading_weight_tensor, heading_weight * heading_proj / 0.8)

    # aligning up axis of robot and environment
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(up_proj > 0.93, up_reward + up_weight, up_reward)

    # energy penalty for movement
    actions_cost = torch.sum(actions**2, dim=-1)
    electricity_cost = torch.sum(
        torch.abs(actions * dof_vel * dof_vel_scale) * motor_effort_ratio.unsqueeze(0),
        dim=-1,
    )

    # dof at limit cost
    dof_at_limit_cost = torch.sum(dof_pos_scaled > 0.98, dim=-1)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * alive_reward_scale
    progress_reward = potentials - prev_potentials

    total_reward = (
        progress_reward
        + alive_reward
        + up_reward
        + heading_reward
        + proximity_reward
        - actions_cost_scale * actions_cost
        - energy_cost_scale * electricity_cost
        - dof_at_limit_cost
    )
    # adjust reward for fallen agents
    total_reward = torch.where(reset_terminated, torch.ones_like(total_reward) * death_cost, total_reward)
    return total_reward
