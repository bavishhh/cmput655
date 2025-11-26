from agent.nonlinear.GreedyAC import GreedyAC
from experiment import Experiment
import gymnasium as gym
import numpy as np

env = gym.make("MountainCarContinuous-v0")
eval_env = gym.make("MountainCarContinuous-v0")

pendulum_env = gym.make("Pendulum-v1")
eval_pendulum_env = gym.make("Pendulum-v1")

agent = GreedyAC(
    num_inputs=pendulum_env.observation_space.shape[0],
    action_space=pendulum_env.action_space,
    gamma=0.99,
    alpha=10.0,
    tau=0.01,
    policy="gaussian",
    target_update_interval=1,
    critic_lr=1e-3,
    actor_lr_scale=1,
    critic_hidden_dim=64,
    actor_hidden_dim=64,
    replay_capacity=int(1e5),
    seed=42,
    batch_size=32,
    rho=0.1,
    num_samples=30,
    betas=(0.9, 0.999),
    env=pendulum_env,
    cuda=False,
    clip_stddev=1000
)

exp = Experiment(
    agent=agent,
    env=pendulum_env, 
    eval_env=eval_pendulum_env, 
    eval_episodes=10,
    total_timesteps=int(1e6),
    eval_interval_timesteps=1000
)

exp.run()

# print(exp.info['eval_episode_rewards'])
# print(exp.info["train_episode_rewards"])
