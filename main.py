from agent.nonlinear.GreedyACGA import GreedyAC
from experimentt import Experiment
import gymnasium as gym
import numpy as np
import sys 
import environment
import wandb
from datetime import datetime

env_config = {
    "env_name": "Pendulum-v0",
    "total_timesteps": 100000,
    "steps_per_episode": 1000,
    "eval_interval_timesteps": 10000,
    "eval_episodes": 10,
    "gamma": 0.99,
    "overwrite_rewards": False,
	"continuous": True,
    "rewards": {},
    "seed": 42,
    "start_state": [],
    "num_grad_steps": 50,
    "ga_lr": 0.01,
}

RANDOM_SEED = env_config["seed"]
monitor = False
after = -1

env = environment.Environment(env_config, RANDOM_SEED, monitor, after)
eval_env = environment.Environment(env_config, RANDOM_SEED)

agent = GreedyAC(
    num_inputs=env.observation_space.shape[0],
    action_space=env.action_space,
    gamma=0.99,
    alpha=0.2,
    tau=0.01,
    policy="gaussian",
    target_update_interval=10,
    critic_lr=1e-3,
    actor_lr_scale=1,
    critic_hidden_dim=64,
    actor_hidden_dim=64,
    replay_capacity=int(1e5),
    seed=RANDOM_SEED,
    batch_size=32,
    rho=0.1,
    num_samples=30,
    betas=(0.9, 0.999),
    env=env,
    cuda=False,
    clip_stddev=1000,
    num_grad_steps=env_config["num_grad_steps"],
    ga_lr=env_config["ga_lr"],
)

exp = Experiment(
    agent=agent,
    env=env,
    eval_env=eval_env,
    eval_episodes=env_config["eval_episodes"],
    total_timesteps=env_config["total_timesteps"],
    eval_interval_timesteps=env_config["eval_interval_timesteps"],
)

wandb.init(
    entity="cmput655",
    project="GARE-GA",
    group=datetime.now().strftime("%y-%m-%d-%H-%M-%S"),
    config=env_config,
)

exp.run()

# print(exp.info['eval_episode_rewards'])
# print(exp.info["train_episode_rewards"])