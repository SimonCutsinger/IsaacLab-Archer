from __future__ import annotations

import torch
import math
import heapq
import random
from pathlib import Path
from collections.abc import Sequence
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from .waypoint import WAYPOINT_CFG
from isaaclab_assets.robots.leatherback import LEATHERBACK_CFG
from isaaclab.markers import VisualizationMarkers
from isaaclab_tasks.direct.archerproject.archer_interactive_scene import ArcherSceneCfg

cwd = Path.cwd()

@configclass
class LeatherbackEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 120
    action_space = 2
    observation_space = 8
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)
    robot_cfg: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    waypoint_cfg = WAYPOINT_CFG

    throttle_dof_name = [
        "Wheel__Knuckle__Front_Left",
        "Wheel__Knuckle__Front_Right",
        "Wheel__Upright__Rear_Right",
        "Wheel__Upright__Rear_Left"
    ]
    steering_dof_name = [
        "Knuckle__Upright__Front_Right",
        "Knuckle__Upright__Front_Left",
    ]

    env_spacing = 30

    scene: ArcherSceneCfg = ArcherSceneCfg()

class LeatherbackEnv(DirectRLEnv):
    cfg: LeatherbackEnvCfg

    def heuristic(self, a, b):
            # Manhattan distance
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def astar(self, occupancy_map, start, goal):
        neighbors = [(0,1),(1,0),(-1,0),(0,-1)]  # 4-directional

        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: self.heuristic(start, goal)}  # Now accessible
        oheap = []

        heapq.heappush(oheap, (fscore[start], start))
        
        while oheap:
            current = heapq.heappop(oheap)[1]

            if current == goal:
                # reconstruct path
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                data.append(start)
                return data[::-1]  # reverse

            close_set.add(current)
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j            
                tentative_g_score = gscore[current] + 1

                if 0 <= neighbor[0] < occupancy_map.shape[1]:
                    if 0 <= neighbor[1] < occupancy_map.shape[0]:
                        if occupancy_map[neighbor[1]][neighbor[0]] == 1:
                            continue
                    else:
                        continue
                else:
                    continue

                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue

                if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)  # Now accessible
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))

        return False

    def _occ_map(self):

        maze_data = []
        marker_positions = []

        with open(f"{cwd}\\source\\isaaclab_tasks\\isaaclab_tasks\\direct\\leatherback\\maze.txt", "r") as file:
            lines = file.readlines()

        for line in lines:
            line = line.strip().rstrip(',')
            parts = line.split(',')
            x, y, z = int(parts[0]), int(parts[1]), int(parts[2])
            rot_x, rot_y, rot_z = int(parts[3]), int(parts[4]), int(parts[5])
            material = parts[6]
            shape = parts[7]
            maze_data.append((x, y, z, rot_x, rot_y, rot_z, material, shape))

        maze_start = 3
        maze_end = 30

        occupancy_map = torch.zeros((maze_end - maze_start, maze_end - maze_start), dtype=torch.uint8)

        for tile in maze_data:
            x, y, z, *_ = tile

            if maze_start <= x < maze_end and maze_start <= z < maze_end and y > 0:
                grid_x = x - maze_start
                grid_z = z - maze_start
                occupancy_map[grid_z, grid_x] = 1 

        entrance_candidates = []
        entrance_sides = []
        maze_min = 0
        maze_max = 26

        # Detect entrances on the maze border
        for x in range(maze_min, maze_max):
            if occupancy_map[maze_min, x] == 0:  # Top edge
                entrance_candidates.append((x, maze_min))
                entrance_sides.append('top')

        for z in range(maze_min, maze_max):
            if occupancy_map[z, maze_min] == 0:  # Left edge
                entrance_candidates.append((maze_min, z))
                entrance_sides.append('left')

        for x in range(maze_min, maze_max):
            if occupancy_map[maze_max, x] == 0:  # Bottom edge
                entrance_candidates.append((x, maze_max))
                entrance_sides.append('bottom')

        for z in range(maze_min, maze_max):
            if occupancy_map[z, maze_max] == 0:  # Right edge
                entrance_candidates.append((maze_max, z))
                entrance_sides.append('right')

        # Choose two entrances
        first_index = random.choice(range(len(entrance_candidates)))
        self.first_spawn = entrance_candidates[first_index]
        self.first_side = entrance_sides[first_index]

        remaining_indices = list(range(len(entrance_candidates)))
        remaining_indices.remove(first_index)

        second_index = random.choice(remaining_indices)
        self.second_spawn = entrance_candidates[second_index]
        self.second_side = entrance_sides[second_index]

        # Find path in the maze 
        path = self.astar(occupancy_map, self.first_spawn, self.second_spawn)

        if path:
            path_offset = [(x + maze_start, z + maze_start) for (x, z) in path]
            marker_positions = torch.tensor(path_offset, dtype=torch.float32, device=self.device)
        else:
            print("No path found!")

        return marker_positions

    def __init__(self, cfg: LeatherbackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._throttle_dof_idx, _ = self.leatherback.find_joints(self.cfg.throttle_dof_name)
        self._steering_dof_idx, _ = self.leatherback.find_joints(self.cfg.steering_dof_name)
        self._throttle_state = torch.zeros((self.num_envs,4), device=self.device, dtype=torch.float32)
        self._steering_state = torch.zeros((self.num_envs,2), device=self.device, dtype=torch.float32)
        self._goal_reached = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        self.task_completed = torch.zeros((self.num_envs), device=self.device, dtype=torch.bool)
        self.env_spacing = self.cfg.env_spacing
        self.position_tolerance = 0.15
        self.goal_reached_bonus = 10.0
        self.position_progress_weight = 1.0
        self.heading_coefficient = 0.25
        self.heading_progress_weight = 0.05
        self._target_index = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        
    def _setup_scene(self):

        # Setup rest of the scene
        self.leatherback = Articulation(self.cfg.robot_cfg)
        self.waypoints = VisualizationMarkers(self.cfg.waypoint_cfg)
        self.object_state = []
        
        self.scene.articulations["leatherback"] = self.leatherback

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        throttle_scale = 10
        throttle_max = 50
        steering_scale = 0.1
        steering_max = 0.85

        self._throttle_action = actions[:, 0].repeat_interleave(4).reshape((-1, 4)) * throttle_scale
        self.throttle_action = torch.clamp(self._throttle_action, -throttle_max, throttle_max)
        self._throttle_state = self._throttle_action
        
        self._steering_action = actions[:, 1].repeat_interleave(2).reshape((-1, 2)) * steering_scale
        self._steering_action = torch.clamp(self._steering_action, -steering_max, steering_max)
        self._steering_state = self._steering_action

    def _apply_action(self) -> None:
        self.leatherback.set_joint_velocity_target(self._throttle_action, joint_ids=self._throttle_dof_idx)
        self.leatherback.set_joint_position_target(self._steering_state, joint_ids=self._steering_dof_idx)

    def _get_observations(self) -> dict:
        current_target_positions = self._target_positions[self.leatherback._ALL_INDICES, self._target_index]
        self._position_error_vector = current_target_positions - self.leatherback.data.root_pos_w[:, :2]
        self._previous_position_error = self._position_error.clone()
        self._position_error = torch.norm(self._position_error_vector, dim=-1)

        heading = self.leatherback.data.heading_w
        target_heading_w = torch.atan2(
            self._target_positions[self.leatherback._ALL_INDICES, self._target_index, 1] - self.leatherback.data.root_link_pos_w[:, 1],
            self._target_positions[self.leatherback._ALL_INDICES, self._target_index, 0] - self.leatherback.data.root_link_pos_w[:, 0],
        )
        self.target_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))

        obs = torch.cat(
            (
                self._position_error.unsqueeze(dim=1),
                torch.cos(self.target_heading_error).unsqueeze(dim=1),
                torch.sin(self.target_heading_error).unsqueeze(dim=1),
                self.leatherback.data.root_lin_vel_b[:, 0].unsqueeze(dim=1),
                self.leatherback.data.root_lin_vel_b[:, 1].unsqueeze(dim=1),
                self.leatherback.data.root_ang_vel_w[:, 2].unsqueeze(dim=1),
                self._throttle_state[:, 0].unsqueeze(dim=1),
                self._steering_state[:, 0].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        
        if torch.any(obs.isnan()):
            raise ValueError("Observations cannot be NAN")

        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        position_progress_rew = self._previous_position_error - self._position_error
        target_heading_rew = torch.exp(-torch.abs(self.target_heading_error) / self.heading_coefficient)
        goal_reached = self._position_error < self.position_tolerance
        self._target_index = self._target_index + goal_reached
        self.task_completed = self._target_index > (self._num_goals -1)
        self._target_index = self._target_index % self._num_goals

        composite_reward = (
            position_progress_rew * self.position_progress_weight +
            target_heading_rew * self.heading_progress_weight +
            goal_reached * self.goal_reached_bonus
        )

        one_hot_encoded = torch.nn.functional.one_hot(self._target_index.long(), num_classes=self._num_goals)
        marker_indices = one_hot_encoded.view(-1).tolist()
        self.waypoints.visualize(marker_indices=marker_indices)

        if torch.any(composite_reward.isnan()):
            raise ValueError("Rewards cannot be NAN")

        return composite_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        task_failed = self.episode_length_buf > self.max_episode_length
        return task_failed, self.task_completed

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.leatherback._ALL_INDICES
        super()._reset_idx(env_ids)

        # Ensure the shape is correct for multiple environments
        marker_positions = self._occ_map()  # Get marker_positions from _occ_map()

        self._num_goals = len(marker_positions)
        self._target_positions = torch.zeros((self.num_envs, self._num_goals, 2), device=self.device, dtype=torch.float32)
        self._markers_pos = torch.zeros((self.num_envs, self._num_goals, 3), device=self.device, dtype=torch.float32)

        self._target_positions[env_ids, :, :] = 0.0
        self._markers_pos[env_ids, :, :] = 0.0

        # Expand manual_positions to handle multiple environments
        manual_positions = marker_positions.unsqueeze(0).expand(len(env_ids), -1, -1)

        # Assign manual positions
        self._target_positions[env_ids, :, :2] = manual_positions

        # Adjust positions based on the environment's origin
        self._target_positions[env_ids, :] += self.scene.env_origins[env_ids, :2].unsqueeze(1)

        self._target_index[env_ids] = 0

        # Define the height for markers (adjust as needed)
        marker_height = 0.5  # Change this value to the desired height

        # Assign X and Y from target positions
        self._markers_pos[env_ids, :, :2] = self._target_positions[env_ids]

        # Set Z to a fixed height
        self._markers_pos[env_ids, :, 2] = marker_height

        # car
        default_state = self.leatherback.data.default_root_state[env_ids]
        leatherback_pose = default_state[:, :7]
        leatherback_velocities = default_state[:, 7:]
        joint_positions = self.leatherback.data.default_joint_pos[env_ids]
        joint_velocities = self.leatherback.data.default_joint_vel[env_ids]

        first_marker = marker_positions[0]  # This will give a tensor of [x, y, z

        if self.first_side == 'top':
            # Rotate car to face upwards (north)
            leatherback_pose[:, 0] = self.scene.env_origins[env_ids, 0] + first_marker[0]      # X position
            leatherback_pose[:, 1] = self.scene.env_origins[env_ids, 1] + first_marker[1] - 1  # Y position
            leatherback_pose[:, 2] = self.scene.env_origins[env_ids, 2] + 1                    # Height (Z)
            yaw = math.radians(90)
        elif self.first_side == 'bottom':
            # Rotate car to face downwards (south)
            leatherback_pose[:, 0] = self.scene.env_origins[env_ids, 0] + first_marker[0]      # X position
            leatherback_pose[:, 1] = self.scene.env_origins[env_ids, 1] + first_marker[1] + 1  # Y position
            leatherback_pose[:, 2] = self.scene.env_origins[env_ids, 2] + 1                    # Height (Z)
            yaw = math.radians(270)
        elif self.first_side == 'left':
            # Rotate car to face left (west)
            leatherback_pose[:, 0] = self.scene.env_origins[env_ids, 0] + first_marker[0] - 1  # X position
            leatherback_pose[:, 1] = self.scene.env_origins[env_ids, 1] + first_marker[1]      # Y position
            leatherback_pose[:, 2] = self.scene.env_origins[env_ids, 2] + 1                    # Height (Z)
            yaw = math.radians(0)
        elif self.first_side == 'right':
            # Rotate car to face right (east)
            leatherback_pose[:, 0] = self.scene.env_origins[env_ids, 0] + first_marker[0] + 1 # X position
            leatherback_pose[:, 1] = self.scene.env_origins[env_ids, 1] + first_marker[1]     # Y position
            leatherback_pose[:, 2] = self.scene.env_origins[env_ids, 2] + 1                   # Height (Z)
            yaw = math.radians(180)

        # Construct quaternion for yaw rotation (only around z-axis here)
        w = math.cos(yaw / 2)
        z = math.sin(yaw / 2)

        # Quaternion components for rotation around the z-axis
        leatherback_pose[:, 3] = w  # w component
        leatherback_pose[:, 4] = 0.0  # x component (no rotation around x-axis)
        leatherback_pose[:, 5] = 0.0  # y component (no rotation around y-axis)
        leatherback_pose[:, 6] = z  # z component
        
        self.leatherback.write_root_pose_to_sim(leatherback_pose, env_ids)
        self.leatherback.write_root_velocity_to_sim(leatherback_velocities, env_ids)
        self.leatherback.write_joint_state_to_sim(joint_positions, joint_velocities, None, env_ids)

        # Visualize updated markers
        visualize_pos = self._markers_pos.view(-1, 3)
        self.waypoints.visualize(translations=visualize_pos)

        current_target_positions = self._target_positions[self.leatherback._ALL_INDICES, self._target_index]
        self._position_error_vector = current_target_positions[:, :2] - self.leatherback.data.root_pos_w[:, :2]
        self._position_error = torch.norm(self._position_error_vector, dim=-1)
        self._previous_position_error = self._position_error.clone()

        heading = self.leatherback.data.heading_w[:]
        target_heading_w = torch.atan2( 
            self._target_positions[:, 0, 1] - self.leatherback.data.root_pos_w[:, 1],
            self._target_positions[:, 0, 0] - self.leatherback.data.root_pos_w[:, 0],
        )
        self._heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))
        self._previous_heading_error = self._heading_error.clone()
        