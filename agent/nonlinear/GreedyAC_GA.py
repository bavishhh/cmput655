# Import modules
from gym.spaces import Box, Discrete
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from agent.baseAgent import BaseAgent
from utils.experience_replay import TorchBuffer as ExperienceReplay
from agent.nonlinear.value_function.MLP import Q as QMLP
from agent.nonlinear.policy.MLP import SquashedGaussian, Gaussian, Softmax
import agent.nonlinear.nn_utils as nn_utils
import inspect
import sys
import wandb

log_to_wandb = False

class GreedyAC_GA(BaseAgent):
    """
    GreedyAC implements the GreedyAC algorithm with continuous actions.
    """
    def __init__(self, num_inputs, action_space, gamma, tau, alpha, policy,
                 target_update_interval, critic_lr, actor_lr_scale,
                 actor_hidden_dim, critic_hidden_dim, replay_capacity, seed,
                 batch_size, betas, env, cuda=False,
                 clip_stddev=1000, init=None, entropy_from_single_sample=True,
                 activation="relu", num_grad_steps=20, ga_lr=0.001):
        super().__init__()

        self.batch = True
        betas = [0.9, 0.999]

        # Ensure batch size < replay capacity
        if batch_size > replay_capacity:
            raise ValueError("cannot have a batch larger than replay " +
                             "buffer capacity")

        # Set the seed for all random number generators, this includes
        # everything used by PyTorch, including setting the initial weights
        # of networks. PyTorch prefers seeds with many non-zero binary units
        self.seed = seed
        self.torch_rng = torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.is_training = True
        self.entropy_from_single_sample = entropy_from_single_sample
        self.gamma = gamma  # Discount factor
        self.tau = tau  # Polyak average
        self.alpha = alpha  # Entropy scale
        self.gradient_ascent = True
        self.gradient_ascent_steps = num_grad_steps
        self.state_dims = num_inputs
        self.discrete_action = isinstance(action_space, Discrete)
        self.action_space = action_space
        self.ga_lr = ga_lr
        self.critic_lr = critic_lr
        self.actor_lr_scale = actor_lr_scale
        self.target_update_interval = target_update_interval

        self.device = torch.device("cuda:0" if cuda and
                                   torch.cuda.is_available() else "cpu")

        if isinstance(action_space, Box):
            self.action_dims = len(action_space.high)

            # Keep a replay buffer
            self.replay = ExperienceReplay(replay_capacity, seed,
                                           env.observation_space.shape,
                                           action_space.shape[0], self.device)
        elif isinstance(action_space, Discrete):
            self.action_dims = 1
            # Keep a replay buffer
            self.replay = ExperienceReplay(replay_capacity, seed,
                                           env.observation_space.shape,
                                           1, self.device)
        self.batch_size = 32
        # Set the interval between timesteps when the target network should be
        # updated and keep a running total of update number
        self.target_update_interval = 1
        self.update_number = 0

        # For GreedyAC update
        self.rho = 1
        self.num_samples = 1

        # Create the critic Q function
        if isinstance(action_space, Box):
            action_shape = action_space.shape[0]
        elif isinstance(action_space, Discrete):
            action_shape = 1


        self.critic = QMLP(num_inputs, action_shape, critic_hidden_dim,
                           init, activation).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_lr,
                                 betas=betas)

        self.critic_target = QMLP(num_inputs, action_shape,
                                  critic_hidden_dim, init, activation).to(
                                          self.device)
        nn_utils.hard_update(self.critic_target, self.critic)

        self._create_policies(policy, num_inputs, action_space,
                              actor_hidden_dim, clip_stddev, init, activation)

        actor_lr = self.actor_lr_scale * self.critic_lr
        self.policy_optim = Adam(self.policy.parameters(), lr=actor_lr,
                                 betas=betas)
        self.sampler_optim = Adam(self.sampler.parameters(), lr=actor_lr,
                                  betas=betas)
        nn_utils.hard_update(self.sampler, self.policy)

        self.is_training = True

        source = inspect.getsource(inspect.getmodule(inspect.currentframe()))
        self.info["source"] = source

    def update(self, state, action, reward, next_state, done_mask):
        # Adjust action shape to ensure it fits in replay buffer properly
        if self.discrete_action:
            action = np.array([action])

        # Keep transition in replay buffer
        self.replay.push(state, action, reward, next_state, done_mask)

        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, \
            mask_batch = self.replay.sample(batch_size=self.batch_size)

        if state_batch is None:
            # Too few samples in the buffer to sample
            return

        # Sample actions from the sampler to determine which to update
        # with. I have changed to 1, so shape will be [batch,1,action_dim]
        sampler_action_batch, _, _, = self.sampler.sample(state_batch,
                                                  self.num_samples)
        # action_batch = action_batch.permute(1, 0, 2)
        sampler_action_batch = sampler_action_batch.reshape(self.batch_size * self.num_samples,
                                            self.action_dims)
        stacked_s_batch = state_batch.repeat_interleave(self.num_samples,
                                                        dim=0)
        stacked_next_state_batch = next_state_batch.repeat_interleave(self.num_samples,
                                                                dim=0)
        # Gradient ascent part
        initial_action_estimates = sampler_action_batch.clone()
        initial_action_estimates.requires_grad_(True)
        optimizer = torch.optim.Adam([initial_action_estimates], lr=self.ga_lr)

        for i in range(self.gradient_ascent_steps):
            optimizer.zero_grad()
            q_values = self.critic(stacked_next_state_batch, initial_action_estimates)
            q_values = q_values.reshape(self.batch_size, self.num_samples, 1)
            q_sum = -1 *  q_values.mean()
            if log_to_wandb and i == 0:
                wandb.log({'GreedyAC/Initial_Avg_Q_Value': -q_sum.item()})
            q_sum.backward()
            optimizer.step()

        if log_to_wandb:
            wandb.log({'GreedyAC/Final_Avg_Q_Value': -q_sum.item()})

        with torch.no_grad():
            # best_actions = initial_action_estimates
            best_actions = torch.clamp(initial_action_estimates.detach(),
                                                   self.action_space.low[0], self.action_space.high[0])

        samples = int(self.rho * self.num_samples)
        stacked_s_batch = state_batch.repeat_interleave(samples, dim=0)
        best_actions = torch.reshape(best_actions, (-1, self.action_dims))
        # Actor loss
        # print(stacked_s_batch.shape, best_actions.shape)
        # print("Computing actor loss")
        policy_loss = self.policy.log_prob(stacked_next_state_batch, best_actions)
        policy_loss = torch.clamp(policy_loss, -50, 50)
        policy_loss = -policy_loss.mean()

        if log_to_wandb:
            wandb.log({'GreedyAC/Policy_Loss': policy_loss.item()})

        # Update actor
        self.policy_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
        self.policy_optim.step()

        # Calculate sampler entropy
        stacked_s_batch = state_batch.repeat_interleave(self.num_samples,
                                                        dim=0)
        stacked_next_state_batch = next_state_batch.repeat_interleave(self.num_samples,
                                                                dim=0)
        stacked_s_batch = stacked_s_batch.reshape(-1, self.state_dims)
        stacked_next_state_batch = stacked_next_state_batch.reshape(-1, self.state_dims)
        action_batch = action_batch.reshape(-1, self.action_dims)

        sampler_entropy = self.sampler.log_prob(stacked_next_state_batch, action_batch)
        sampler_entropy = torch.clamp(sampler_entropy, -50, 50)
        # with torch.no_grad():
        #     sampler_entropy *= sampler_entropy

        sampler_entropy = sampler_entropy.reshape(self.batch_size,
                                                  self.num_samples, 1)
        if self.entropy_from_single_sample:
            sampler_entropy = -sampler_entropy[:, 0, :]
        else:
            sampler_entropy = -sampler_entropy.mean(axis=1)

        # Calculate sampler loss

        stacked_next_state_batch = next_state_batch.repeat_interleave(samples, dim=0)
        sampler_loss = self.sampler.log_prob(stacked_next_state_batch, best_actions)
        sampler_loss = torch.clamp(sampler_loss, -50, 50)
        sampler_loss = sampler_loss.reshape(self.batch_size, samples, 1)
        sampler_loss = sampler_loss.mean(axis=1)
        sampler_loss = sampler_loss + (sampler_entropy * self.alpha)
        sampler_loss = -sampler_loss.mean()
        if log_to_wandb:   
            wandb.log({'GreedyAC/Sampler_Loss': sampler_loss.item()})

        # Update the sampler
        self.sampler_optim.zero_grad()
        sampler_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.sampler.parameters(), max_norm=10.0)
        self.sampler_optim.step()

        # next_state_action, _, _ = self.policy.sample(next_state_batch)
        with torch.no_grad():
            next_q = self.critic_target(next_state_batch, best_actions)
            target_q_value = reward_batch + mask_batch * self.gamma * next_q

        q_value = self.critic(state_batch, action_batch)

        # Calculate the loss on the critic
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        q_loss = F.mse_loss(target_q_value, q_value)
        if log_to_wandb:    
            wandb.log({'GreedyAC/Critic_Loss': q_loss.item()})

        # Update the critic
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        # Update target networks
        self.update_number += 1
        if self.update_number % self.target_update_interval == 0:
            self.update_number = 0
            nn_utils.soft_update(self.critic_target, self.critic, self.tau)

    def sample_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if self.is_training:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)

        act = action.detach().cpu().numpy()[0]

        if not self.discrete_action:
            return act
        else:
            return int(act[0])

    def reset(self):
        pass

    def eval(self):
        self.is_training = False

    def train(self):
        self.is_training = True

    def _create_policies(self, policy, num_inputs, action_space,
                         actor_hidden_dim, clip_stddev, init, activation):
        self.policy_type = policy.lower()
        if self.policy_type == "gaussian":
            self.policy = Gaussian(num_inputs, action_space.shape[0],
                                   actor_hidden_dim, activation,
                                   action_space, clip_stddev,
                                   init).to(self.device)

            self.sampler = Gaussian(num_inputs, action_space.shape[0],
                                    actor_hidden_dim, activation,
                                    action_space, clip_stddev,
                                    init).to(self.device)

        elif self.policy_type == "squashedgaussian":
            self.policy = SquashedGaussian(num_inputs, action_space.shape[0],
                                           actor_hidden_dim, activation,
                                           action_space, clip_stddev,
                                           init).to(self.device)

            self.sampler = SquashedGaussian(num_inputs, action_space.shape[0],
                                            actor_hidden_dim, activation,
                                            action_space, clip_stddev,
                                            init).to(self.device)

        elif self.policy_type == "softmax":
            num_actions = action_space.n
            self.policy = Softmax(num_inputs, num_actions,
                                  actor_hidden_dim, activation,
                                  action_space, init).to(self.device)

            self.sampler = Softmax(num_inputs, num_actions,
                                   actor_hidden_dim, activation,
                                   action_space, init).to(self.device)

        else:
            raise NotImplementedError

    def get_parameters(self):
        pass

    def save_model(self, env_name, suffix="", actor_path=None,
                   critic_path=None):
        pass

    def load_model(self, actor_path, critic_path):
        pass