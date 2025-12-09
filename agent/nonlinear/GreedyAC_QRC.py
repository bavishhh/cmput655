# Import modules
from gym.spaces import Box, Discrete
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from agent.baseAgent import BaseAgent
from utils.experience_replay import TorchBuffer as ExperienceReplay
from agent.nonlinear.value_function.MLP import SharedQ
from agent.nonlinear.policy.MLP import SquashedGaussian, Gaussian, Softmax
import agent.nonlinear.nn_utils as nn_utils
import inspect


class GreedyAC_QRC(BaseAgent):
    """
    GreedyAC implements the GreedyAC algorithm with continuous actions
    and QRC (Q-Learning with Regularized Corrections) updates.
    """
    def __init__(self, num_inputs, action_space, gamma, tau, alpha, policy,
                 target_update_interval, critic_lr, actor_lr_scale,
                 actor_hidden_dim, critic_hidden_dim, replay_capacity, seed,
                 batch_size, rho, num_samples, betas, env, cuda=False,
                 clip_stddev=1000, init=None, entropy_from_single_sample=True,
                 activation="relu", qrc_beta=1.0): # Added qrc_beta
        super().__init__()

        self.batch = True
        if batch_size > replay_capacity:
            raise ValueError("cannot have a batch larger than replay " +
                             "buffer capacity")

        self.torch_rng = torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.is_training = True
        self.entropy_from_single_sample = entropy_from_single_sample
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.qrc_beta = qrc_beta  # Regularization coefficient for H-network
        self.state_dims = num_inputs
        self.discrete_action = isinstance(action_space, Discrete)
        self.action_space = action_space

        self.device = torch.device("cuda:0" if cuda and
                                   torch.cuda.is_available() else "cpu")

        if isinstance(action_space, Box):
            self.action_dims = len(action_space.high)
            self.replay = ExperienceReplay(replay_capacity, seed,
                                           env.observation_space.shape,
                                           action_space.shape[0], self.device)
        elif isinstance(action_space, Discrete):
            self.action_dims = 1
            self.replay = ExperienceReplay(replay_capacity, seed,
                                           env.observation_space.shape,
                                           1, self.device)
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.update_number = 0
        self.rho = rho
        self.num_samples = num_samples

        if isinstance(action_space, Box):
            action_shape = action_space.shape[0]
        elif isinstance(action_space, Discrete):
            action_shape = 1

        self.critic = SharedQ(num_inputs, action_shape, critic_hidden_dim,
                              init, activation).to(device=self.device)
        
        # Helper to identify H-parameters for regularization later
        # We store the *names* or *references* to distinguish them in the loop
        self.h_param_names = set([n for n, p in self.critic.named_parameters() if "linear4" in n])

        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr, betas=betas)

        self.critic_target = SharedQ(num_inputs, action_shape,
                                     critic_hidden_dim, init, activation).to(
                                          self.device)
        nn_utils.hard_update(self.critic_target, self.critic)

        self._create_policies(policy, num_inputs, action_space,
                              actor_hidden_dim, clip_stddev, init, activation)

        actor_lr = actor_lr_scale * critic_lr
        self.policy_optim = Adam(self.policy.parameters(), lr=actor_lr,
                                 betas=betas)
        self.sampler_optim = Adam(self.sampler.parameters(), lr=actor_lr,
                                  betas=betas)
        nn_utils.hard_update(self.sampler, self.policy)
        
        self.is_training = True
        source = inspect.getsource(inspect.getmodule(inspect.currentframe()))
        self.info["source"] = source
        
    def update(self, state, action, reward, next_state, done_mask):
        if self.discrete_action:
            action = np.array([action])

        self.replay.push(state, action, reward, next_state, done_mask)
        state_batch, action_batch, reward_batch, next_state_batch, \
            mask_batch = self.replay.sample(batch_size=self.batch_size)

        if state_batch is None:
            return

        # --------------------------------------------------------
        # 1. Action Selection (CEM) on Next State
        # --------------------------------------------------------
        # A. Sample N candidates per next_state
        # next_action_batch shape: (Batch, N, ActionDim)
        next_action_batch, _, _, = self.sampler.sample(next_state_batch, self.num_samples)

        # B. Evaluate candidates using Target Critic
        # We use Target Critic for selection to ensure stability in the target calculation
        flat_next_actions = next_action_batch.reshape(-1, self.action_dims) # (Batch*N, Dim)
        flat_next_states = next_state_batch.repeat_interleave(self.num_samples, dim=0)

        with torch.no_grad():
            # critic_target returns (q, h), we only need q
            target_q_values, _ = self.critic_target(flat_next_states, flat_next_actions)

        # C. Sort to find best actions
        # Reshape to (Batch, N)
        target_q_values = target_q_values.reshape(self.batch_size, self.num_samples)
        
        # Sort descending
        sorted_q, sorted_indices = torch.argsort(target_q_values, dim=1, descending=True)

        # D. Extract Key Values for Updates
        
        # 1. Max Q-value (for Q-Target calculation)
        # Top-1 value for each batch item
        max_target_q = sorted_q[:, 0].unsqueeze(1) # (Batch, 1)

        # 2. Single Best Action a* (for QRC Correction Term)
        # Top-1 index
        best_idx = sorted_indices[:, 0].reshape(self.batch_size, 1, 1).repeat(1, 1, self.action_dims)
        max_action_batch = torch.gather(next_action_batch, 1, best_idx).squeeze(1) # (Batch, Dim)

        # 3. Top Rho Actions (for Actor/Sampler Targets)
        rho_k = int(self.rho * self.num_samples)
        top_rho_indices = sorted_indices[:, :rho_k].unsqueeze(-1).repeat(1, 1, self.action_dims)
        # Gather top rho actions: (Batch, rho_k, Dim)
        top_actions = torch.gather(next_action_batch, 1, top_rho_indices)
        
        # --------------------------------------------------------
        # 2. QRC Update (Critic)
        # --------------------------------------------------------
        self.critic_optim.zero_grad()
        
        # Forward pass on Current Network
        q_pred, h_pred = self.critic(state_batch, action_batch)
        
        # Delta calculation: r + gamma * max_a Q_target(s', a) - Q(s, a)
        target_val = reward_batch + mask_batch * self.gamma * max_target_q
        delta = target_val - q_pred.detach()
        
        params = list(self.critic.named_parameters())
        param_tensors = [p for n, p in params]
        
        # A. Standard Q Gradient: delta * \nabla Q(s, a)
        grads_q = torch.autograd.grad(q_pred, param_tensors, grad_outputs=delta, 
                                      retain_graph=True, allow_unused=True)
        
        # B. Correction Gradient: -gamma * h(s) * \nabla Q(s', a*)
        # We use max_action_batch (a*) derived from CEM above
        q_next_pred_star, _ = self.critic(next_state_batch, max_action_batch)
        correction_scale = -1.0 * self.gamma * h_pred.detach()
        
        grads_correction = torch.autograd.grad(q_next_pred_star, param_tensors, 
                                               grad_outputs=correction_scale, 
                                               retain_graph=True, allow_unused=True)

        # C. H-Function Update: (delta - h) * \nabla h
        h_err = delta - h_pred
        grads_h = torch.autograd.grad(h_pred, param_tensors, grad_outputs=h_err, 
                                      retain_graph=False, allow_unused=True)

        # Apply Gradients & Regularization
        for i, (name, param) in enumerate(params):
            total_grad = torch.zeros_like(param)
            
            if grads_q[i] is not None: total_grad += grads_q[i]
            if grads_correction[i] is not None: total_grad += grads_correction[i]
            if grads_h[i] is not None: total_grad += grads_h[i]
            
            # Regularization for H-network head
            if "linear4" in name:
                total_grad -= self.qrc_beta * param.data

            param.grad = -total_grad

        self.critic_optim.step()

        # Update Target Network
        self.update_number += 1
        if self.update_number % self.target_update_interval == 0:
            self.update_number = 0
            nn_utils.soft_update(self.critic_target, self.critic, self.tau)

        # --------------------------------------------------------
        # 3. Actor / Sampler Update (Reusing CEM results)
        # --------------------------------------------------------
        # We reuse 'top_actions' and 'next_state_batch'
        
        flat_top_actions = top_actions.reshape(-1, self.action_dims)
        stacked_ns_batch_rho = next_state_batch.repeat_interleave(rho_k, dim=0)

        # Actor Loss
        # Maximize log_prob of the top actions found via CEM
        policy_loss = -self.policy.log_prob(stacked_ns_batch_rho, flat_top_actions).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Sampler Loss
        # 1. Entropy Term (Fresh sampling for clean distribution calculation)
        s_action_batch, _, _ = self.sampler.sample(next_state_batch, self.num_samples)
        s_action_batch = s_action_batch.reshape(-1, self.action_dims)
        full_ns_batch = next_state_batch.repeat_interleave(self.num_samples, dim=0)
        
        sampler_entropy = self.sampler.log_prob(full_ns_batch, s_action_batch)
        with torch.no_grad():
            sampler_entropy *= sampler_entropy
        sampler_entropy = sampler_entropy.reshape(self.batch_size, self.num_samples, 1)
        
        if self.entropy_from_single_sample:
             sampler_entropy = -sampler_entropy[:, 0, :]
        else:
             sampler_entropy = -sampler_entropy.mean(axis=1)

        # 2. Likelihood Term (Targeting top_actions)
        sampler_loss = self.sampler.log_prob(stacked_ns_batch_rho, flat_top_actions)
        sampler_loss = sampler_loss.reshape(self.batch_size, rho_k, 1).mean(axis=1)
        
        sampler_loss = sampler_loss + (sampler_entropy * self.alpha)
        sampler_loss = -sampler_loss.mean()

        self.sampler_optim.zero_grad()
        sampler_loss.backward()
        self.sampler_optim.step()

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
    
    # ... (Include helper methods _create_policies, save_model, etc. as defined before)
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
            
    def get_parameters(self): pass
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None): pass
    def load_model(self, actor_path, critic_path): pass