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


class GARE(BaseAgent):
    """
    GreedyAC with QRC updates and Gradient Ascent action refinement.
    """
    def __init__(self, num_inputs, action_space, gamma, tau, alpha, policy,
                 target_update_interval, critic_lr, actor_lr_scale,
                 actor_hidden_dim, critic_hidden_dim, replay_capacity, seed,
                 batch_size, betas, env, cuda=False,
                 clip_stddev=1000, init=None, entropy_from_single_sample=True,
                 activation="relu", qrc_beta=1.0, 
                 ga_lr=0.01, num_grad_steps=5): # Added GA params
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
        self.qrc_beta = qrc_beta
        # Gradient Ascent Parameters
        self.ga_lr = ga_lr
        self.gradient_ascent_steps = num_grad_steps
        
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
        self.rho = 1
        self.num_samples = 1

        if isinstance(action_space, Box):
            action_shape = action_space.shape[0]
        elif isinstance(action_space, Discrete):
            action_shape = 1

        self.critic = SharedQ(num_inputs, action_shape, critic_hidden_dim,
                              init, activation).to(device=self.device)
        
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

    def _find_best_actions(self, state_batch):
        """
        Samples actions and refines them using Gradient Ascent 
        on the Q-function.
        """
        # 1. Sample N actions per state
        # shape: (Batch, N, ActionDim)
        initial_action_estimates, _, _ = self.sampler.sample(state_batch, self.num_samples)
        
        # Prepare for GA
        # Flatten state: (Batch * N, StateDim)
        stacked_state_batch = state_batch.repeat_interleave(self.num_samples, dim=0)
        # Flatten action: (Batch * N, ActionDim)
        initial_action_estimates = initial_action_estimates.reshape(
            self.batch_size * self.num_samples, self.action_dims
        )

        initial_action_estimates.requires_grad_(True)
        optimizer = torch.optim.Adam([initial_action_estimates], lr=self.ga_lr)

        # 2. Gradient Ascent Loop
        for i in range(self.gradient_ascent_steps):
            optimizer.zero_grad()
            
            # Forward pass through SharedQ
            # returns (q, h), we only optimize q
            q_values, _ = self.critic(stacked_state_batch, initial_action_estimates)
            
            q_values = q_values.reshape(self.batch_size, self.num_samples, 1)
            
            # Maximizing Q is minimizing -Q
            q_sum = -1 * q_values.mean()
            q_sum.backward()
            optimizer.step()

        # 3. Clamp and Detach
        with torch.no_grad():
            best_actions = torch.clamp(initial_action_estimates.detach(),
                                       self.action_space.low[0], 
                                       self.action_space.high[0])
            
        # Reshape back to (Batch, N, ActionDim)
        best_actions = best_actions.reshape(self.batch_size, self.num_samples, self.action_dims)
        
        return best_actions, stacked_state_batch

    def update(self, state, action, reward, next_state, done_mask):
        if self.discrete_action:
            action = np.array([action])

        self.replay.push(state, action, reward, next_state, done_mask)
        state_batch, action_batch, reward_batch, next_state_batch, \
            mask_batch = self.replay.sample(batch_size=self.batch_size)

        if state_batch is None:
            return

        # --------------------------------------------------------
        # 1. Refine Actions for Next State (Target Calculation)
        # --------------------------------------------------------
        # We perform the expensive GA step ONCE here.
        # best_next_actions: (Batch, N, ActionDim)
        best_next_actions, _ = self._find_best_actions(next_state_batch)
        
        # Prepare for evaluation
        flat_best_next_actions = best_next_actions.reshape(-1, self.action_dims)
        flat_next_states = next_state_batch.repeat_interleave(self.num_samples, dim=0)

        # Evaluate with Target Critic to find the "value" of these actions
        # We will reuse these values later for sorting in the Actor update
        with torch.no_grad():
            target_q_values, _ = self.critic_target(flat_next_states, flat_best_next_actions)
        
        # Reshape to (Batch, N)
        target_q_values = target_q_values.reshape(self.batch_size, self.num_samples)
        
        # Find Max for Critic Target (QRC requirement)
        max_target_q, argmax_indices = target_q_values.max(dim=1, keepdim=True) # (Batch, 1)
        
        # --------------------------------------------------------
        # 2. QRC Update (Critic)
        # --------------------------------------------------------
        self.critic_optim.zero_grad()
        
        q_pred, h_pred = self.critic(state_batch, action_batch)
        
        target_val = reward_batch + mask_batch * self.gamma * max_target_q
        delta = target_val - q_pred.detach()
        
        params = list(self.critic.named_parameters())
        param_tensors = [p for n, p in params]
        
        # A. Standard Q Gradient
        grads_q = torch.autograd.grad(q_pred, param_tensors, grad_outputs=delta, 
                                      retain_graph=True, allow_unused=True)
        
        # B. Correction Gradient
        # Gather the specific single best action that maximized the Q-value
        argmax_expanded = argmax_indices.unsqueeze(-1).repeat(1, 1, self.action_dims)
        max_action_batch = torch.gather(best_next_actions, 1, argmax_expanded).squeeze(1)
        
        q_next_pred_star, _ = self.critic(next_state_batch, max_action_batch)
        correction_scale = -1.0 * self.gamma * h_pred.detach()
        
        grads_correction = torch.autograd.grad(q_next_pred_star, param_tensors, 
                                               grad_outputs=correction_scale, 
                                               retain_graph=True, allow_unused=True)

        # C. H-Function Update
        h_err = delta - h_pred
        grads_h = torch.autograd.grad(h_pred, param_tensors, grad_outputs=h_err, 
                                      retain_graph=False, allow_unused=True)

        # Apply Gradients with Regularization
        for i, (name, param) in enumerate(params):
            total_grad = torch.zeros_like(param)
            if grads_q[i] is not None: total_grad += grads_q[i]
            if grads_correction[i] is not None: total_grad += grads_correction[i]
            if grads_h[i] is not None: total_grad += grads_h[i]
            
            if "linear4" in name:
                total_grad -= self.qrc_beta * param.data

            param.grad = -total_grad

        self.critic_optim.step()

        self.update_number += 1
        if self.update_number % self.target_update_interval == 0:
            self.update_number = 0
            nn_utils.soft_update(self.critic_target, self.critic, self.tau)

        # --------------------------------------------------------
        # 3. Actor / Sampler Update (Using RECYCLED next_state data)
        # --------------------------------------------------------
        # We reuse 'best_next_actions' and 'target_q_values' calculated in Step 1.
        # This updates the actor to produce best actions for 'next_state_batch'.
        
        # Sort the reused actions based on their Target Q-values
        # (Batch, N)
        sorted_ind = torch.argsort(target_q_values, dim=1, descending=True)
        
        # Select top rho * N
        rho_k = int(self.rho * self.num_samples)
        
        # 1. Get the 2D indices (Batch, rho_k)
        top_ind_base = sorted_ind[:, :rho_k]
        
        # 2. Unsqueeze to add the missing dimension -> (Batch, rho_k, 1)
        top_ind_3d = top_ind_base.unsqueeze(-1)
        
        # 3. Expand to match ActionDim -> (Batch, rho_k, ActionDim)
        top_ind_3d = top_ind_3d.expand(-1, -1, self.action_dims)
        
        # 4. Now gather works because both are 3D
        # best_next_actions: (Batch, N, ActionDim)
        # top_ind_3d:        (Batch, rho_k, ActionDim)
        top_actions = torch.gather(best_next_actions, 1, top_ind_3d)
        
        top_actions_flat = top_actions.reshape(-1, self.action_dims)
        
        # Expand next_state_batch for the subset (since we are using next_states)
        stacked_ns_batch_rho = next_state_batch.repeat_interleave(rho_k, dim=0)

        # Actor Loss
        # We maximize the likelihood of these "top" actions
        policy_loss = -self.policy.log_prob(stacked_ns_batch_rho, top_actions_flat).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Sampler Loss
        # 1. Entropy Term: We need to calc entropy for the distribution at next_state
        # We can just sample freshly from the sampler for entropy calculation to be safe/clean
        # (This is cheap compared to the GA loop)
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

        # 2. Likelihood Term: Maximize log_prob of the 'top_actions' (from Step 1)
        sampler_loss = self.sampler.log_prob(stacked_ns_batch_rho, top_actions_flat)
        sampler_loss = sampler_loss.reshape(self.batch_size, rho_k, 1).mean(axis=1)
        
        # Combine
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