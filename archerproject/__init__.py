import gymnasium as gym

from isaaclab_tasks.direct.cartpole import agents

##
# Register Gym environments.
##

gym.register(
    id="Archer_Cartpole_Direct_v0",
    entry_point=f"{__name__}.archer_env_RLcartpole:CartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.archer_cartpole_env:CartpoleEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)