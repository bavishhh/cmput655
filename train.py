"""
This file contains the training loop for the ActorExpert model.
"""

# Import necessary libraries
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import argparse
import random
from actorexpert import GaussianActorExpert, UniformActorExpert, LaplacianActorExpert

# Define the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str, default="Pendulum-v1", help="Name of the Gym environment")
parser.add_argument("--actor_num_layers", type=int, default=2, help="Number of layers in the actor network")
parser.add_argument("--critic_num_layers", type=int, default=2, help="Number of layers in the critic network")
parser.add_argument("--actor_hidden_layer_sizes", type=int, nargs='+', default=[128, 128], help="Sizes of hidden layers in the actor network")
parser.add_argument("--critic_hidden_layer_sizes", type=int, nargs='+', default=[128, 128], help="Sizes of hidden layers in the critic network")
parser.add_argument("--actor_lr", type=float, default=1e-3, help="Learning rate for the Actor optimizer")
parser.add_argument("--critic_lr", type=float, default=1e-3, help="Learning rate for Critic optimizer")
parser.add_argument("--max_lr", type=float, default=1e-3, help="Learning rate for Q-value maximisation GD")
parser.add_argument("--max_timesteps", type=int, default=200, help="Maximum timesteps per episode")
parser.add_argument("--training_steps", type=int, default=int(1e5), help="Number of total training steps")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--policy_type", type=str, default="gaussian", choices=["gaussian", "uniform", "laplacian"], help="Type of policy parameterization")
parser.add_argument("--alpha", type=float, default=0.1, help="Entropy regularization coefficient")
parser.add_argument("--beta", type=float, default=0.1, help="Regularization coefficient for critic updates")
parser.add_argument("--tau", type=int, default=10, help="Number of gradient steps for Q-value maximization")
args = parser.parse_args()

# Define the device to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters - Given
ENV_NAME = args.env_name
ACTOR_NUM_LAYERS = args.actor_num_layers
CRITIC_NUM_LAYERS = args.critic_num_layers
ACTOR_HIDDEN_LAYER_SIZES = args.actor_hidden_layer_sizes
CRITIC_HIDDEN_LAYER_SIZES = args.critic_hidden_layer_sizes
ACTOR_LR = args.actor_lr
CRITIC_LR = args.critic_lr
MAX_LR = args.max_lr
MAX_TIMESTEPS = args.max_timesteps
BATCH_SIZE = args.batch_size
POLICY_TYPE = args.policy_type
TRAINING_STEPS = args.training_steps
ALPHA = args.alpha
BETA = args.beta
TAU = args.tau

# Hyperparameters - Design Choice
GAMMA = 0.99  
INIT_EXPLORE_FRACTION = 0.01  # Fraction of total training steps to use for initial exploration with random actions
EPSILON = 1e-6
NUM_SEEDS = 30
SEEDS = [random.randint(0, 100000) for _ in range(NUM_SEEDS)]
EVAL_EPISODES = 10

# Create the environment 
env = gym.make(ENV_NAME)
eval_env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_high = env.action_space.high[0]
action_low = env.action_space.low[0]
action_space = env.action_space
print(f"Environment: {ENV_NAME}, State Dim: {state_dim}, Action Dim: {action_dim}")

# Initialize the ActorExpert model based on policy type
policy_types = {
    "gaussian": GaussianActorExpert,
    "uniform": UniformActorExpert,
    "laplacian": LaplacianActorExpert
}

# Initialize the ActorExpert model
model = policy_types[POLICY_TYPE](
    state_dim=state_dim,
    action_dim=action_dim,
    actor_num_layers=ACTOR_NUM_LAYERS,
    actor_hidden_layer_sizes=ACTOR_HIDDEN_LAYER_SIZES,
    critic_num_layers=CRITIC_NUM_LAYERS,
    critic_hidden_layer_sizes=CRITIC_HIDDEN_LAYER_SIZES,
).to(device)

shared_policy, mean_policy, std_policy = model.get_actor_params()
shared_critic, critic_head, td_est_head = model.get_critic_params()

# Define loss functions for each policy parameterization
def gaussian_actor_loss(model, states, improved_actions, alpha=ALPHA):
    (mean, std), _ = model(states)
    dist = torch.distributions.Normal(mean, std)
    log_probs = dist.log_prob(improved_actions).sum(dim=-1, keepdim=True)
    # _, q_values = model(states, improved_actions)
    loss = (-alpha * log_probs).mean()
    return loss

def uniform_actor_loss(model, states, improved_actions, alpha=ALPHA):
    (mean, half_range), _ = model(states)
    lower_bound = mean - half_range
    upper_bound = mean + half_range
    in_bounds = ((improved_actions >= lower_bound) & (improved_actions <= upper_bound)).float()
    log_probs = torch.log(in_bounds / (2 * half_range + EPSILON)).sum(dim=-1, keepdim=True)
    # _, q_values = model(states, improved_actions)
    loss = (-alpha * log_probs).mean()
    return loss

def laplacian_actor_loss(model, states, improved_actions, alpha=ALPHA):
    (mean, b), _ = model(states)
    dist = torch.distributions.Laplace(mean, b)
    log_probs = dist.log_prob(improved_actions).sum(dim=-1, keepdim=True)
    # _, q_values = model(states, improved_actions)
    loss = (-alpha * log_probs).mean()
    return loss

loss_types = {
    "gaussian": gaussian_actor_loss,
    "uniform": uniform_actor_loss,
    "laplacian": laplacian_actor_loss
}

actor_loss = loss_types[POLICY_TYPE]

# Define function to get distribution
def get_dist(mean, param, policy_type):
    if policy_type == "gaussian":
        dist = torch.distributions.Normal(mean, param)
    elif policy_type == "uniform":
        lower_bound = mean - param
        upper_bound = mean + param
        dist = torch.distributions.Uniform(lower_bound, upper_bound)
    elif policy_type == "laplacian":
        dist = torch.distributions.Laplace(mean, param)
    return dist

# Define the optimizers [Learning rates of critic and td_est are same as design choice (Patterson, 2021)]
actor_optimizer = optim.Adam(list(shared_policy) + list(mean_policy) + list(std_policy), lr=ACTOR_LR)
critic_optimizer = optim.Adam(list(shared_critic) + list(critic_head), lr=CRITIC_LR)
td_est_optimizer = optim.Adam(list(td_est_head), lr=CRITIC_LR)

# Create a Replay Buffer class 
class ReplayBuffer:
    def __init__(self, max_size=int(1e5), batch_size=64):
        self.buffer = []
        self.max_size = max_size
        self.batch_size = batch_size
        self.ptr = 0
    
    def __len__(self):
        return len(self.buffer)

    def add(self, experience):
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.ptr] = experience
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self):
        indices = np.random.choice(len(self.buffer), min(self.batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return (np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones))

replay_buffer = ReplayBuffer()

# Store the average returns along training
average_returns = np.zeros(TRAINING_STEPS)

# Define a function to calculate TD targets 
def compute_td_target(model, rewards, next_states, dones, gamma=GAMMA):
    with torch.no_grad():
        # Sample next actions from the current policy
        (next_mean, next_param), _ = model(next_states)
        dist = get_dist(next_mean, next_param, POLICY_TYPE)
        next_actions = dist.rsample()
    # Perform Gradient Ascent to find maximum Q-value for next states without updating model parameters
    saved_state = {k: v.clone() for k, v in model.state_dict().items()}
    def loss_fn(next_state, next_action):
        _, q_value = model(next_state, next_action)
        return -q_value
    next_actions.requires_grad = True
    next_states.requires_grad = True
    def f(action):
        return loss_fn(next_states, action).squeeze(-1)
    improved_next_actions = next_actions.clone()
    for _ in range(TAU):
        # TODO: Implement gradient ascent step
        J = torch.autograd.functional.jacobian(f, improved_next_actions)
        action_grad = torch.stack([J[i, i] for i in range(improved_next_actions.shape[0])], dim=0)
        improved_next_actions = improved_next_actions + MAX_LR * action_grad
        # Clip actions to be within valid bounds
        improved_next_actions = torch.clamp(improved_next_actions, action_low, action_high) 
    # Restore model parameters
    model.load_state_dict(saved_state)
    _, next_q_values = model(next_states, improved_next_actions)
    td_targets = rewards + gamma * (1 - dones) * next_q_values
    return td_targets, next_actions, improved_next_actions

# Define a regularised loss for TD estimate
def compute_td_est_loss(model, states, actions, td_targets, beta=BETA):
    td_ests = model(states, actions, td_est=True)[1]
    mse_loss = F.mse_loss(td_ests, td_targets)
    reg_loss = 0.0
    for param in model.get_critic_params()[-1]:
        reg_loss += torch.sum(param ** 2)
    loss = mse_loss + beta * reg_loss
    return loss

# Define function to compute critic gradients
def compute_critic_gradients(model, states, actions, next_states, improved_next_actions):
    _, current_q_values = model(states, actions)        # [B,1]
    _, next_q_values    = model(next_states, improved_next_actions)
    current_q_values = current_q_values.squeeze(-1)     # [B]
    next_q_values    = next_q_values.squeeze(-1)
    params = list(model.get_critic_params()[0]) + list(model.get_critic_params()[1])
    current_q_gradients = []
    next_q_gradients = []
    for p in params:
        batch_cgrads = []
        batch_qgrads = []
        for i in range(current_q_values.shape[0]):
            cgrads = torch.autograd.grad(current_q_values[i], p, retain_graph=True, create_graph=True)[0]
            batch_cgrads.append(cgrads.unsqueeze(0))
            qgrads = torch.autograd.grad(next_q_values[i], p, retain_graph=True, create_graph=True)[0]
            batch_qgrads.append(qgrads.unsqueeze(0))
        batch_cgrads = torch.cat(batch_cgrads, dim=0)  # [B, *p.shape]
        batch_qgrads = torch.cat(batch_qgrads, dim=0)  # [B, *p.shape]
        current_q_gradients.append(batch_cgrads)
        next_q_gradients.append(batch_qgrads)
    return current_q_gradients, next_q_gradients

# Offline Evaluation Function
def evaluate_policy(model, eval_env, episodes=EVAL_EPISODES):
    with torch.no_grad():
        total_return = 0.0
        for _ in range(episodes):
            state, _ = eval_env.reset()
            episode_return = 0.0
            done = False
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                (mean, param) = model(state_tensor)[0]
                dist = get_dist(mean, param, POLICY_TYPE)
                action = dist.mean.cpu().detach().numpy()[0]  # Use mean action for evaluation
                action = np.clip(action, action_low, action_high)
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                episode_return += reward
                state = next_state
            total_return += episode_return
        average_return = total_return / episodes
    return average_return

# Training Loop
state, _ = env.reset()
episode_start = 0
episode_returns = []
offline_episode_returns = []
episode_reward = 0

for training_step in range(TRAINING_STEPS):
    if training_step < int(INIT_EXPLORE_FRACTION * TRAINING_STEPS):
        # Take random actions for initial exploration
        action = env.action_space.sample()
    else:
        # Sample action from the current policy
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        (mean, param) = model(state_tensor)[0]
        dist = get_dist(mean, param, POLICY_TYPE)
        action = dist.sample().cpu().detach().numpy()[0]
        action = np.clip(action, action_low, action_high)
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    replay_buffer.add((state, action, reward, next_state, done))

    # Calculate average returns
    if training_step == episode_start:
        average_returns[training_step] = reward
    else:
        average_returns[training_step] = average_returns[training_step - 1] + (reward - average_returns[training_step - 1]) / (training_step - episode_start + 1)
    state = next_state
    episode_reward += reward
    if done:
        state, _ = env.reset()
        episode_start = training_step + 1
        episode_returns.append(episode_reward)
        episode_reward = 0

    # Sample a batch from the replay buffer
    states, actions, rewards, next_states, dones = replay_buffer.sample()
    states = torch.tensor(states, dtype=torch.float32, requires_grad=True).to(device)
    actions = torch.tensor(actions, dtype=torch.float32, requires_grad=True).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32, requires_grad=True).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
    training_step += 1
    # Skip the update steps if its the initial exploration phase
    if training_step >= int(INIT_EXPLORE_FRACTION * TRAINING_STEPS):
        # Update Critic Network (TODO: Implement critic update logic)
        critic_optimizer.zero_grad()
        td_est_optimizer.zero_grad()
        actor_optimizer.zero_grad()
        # Compute TD targets
        td_targets, next_actions, improved_next_actions = compute_td_target(model, rewards, next_states, dones)
        # Compute critic gradients and td_est loss
        _, q_values, td_ests = model(states, actions, td_est=True)
        current_q_gradient, next_q_gradient = compute_critic_gradients(model, states, actions, next_states, improved_next_actions)
        index = 0
        for param in list(model.get_critic_params()[0]) + list(model.get_critic_params()[1]):
            cgrad, qgrad = current_q_gradient[index], next_q_gradient[index]
            if len(param.shape) > 1:
                td_ests = td_ests.unsqueeze(-1)
                td_targets = td_targets.unsqueeze(-1)
            term1 = GAMMA * td_ests * qgrad
            term2 = td_targets * cgrad
            param_grad = term1.sum(dim=0) - term2.sum(dim=0)
            if len(param.shape) > 1:
                td_ests = td_ests.squeeze(-1)
                td_targets = td_targets.squeeze(-1)
            index += 1
        td_est_loss = compute_td_est_loss(model, states, actions, td_targets)
        td_est_loss.backward()
        critic_optimizer.step()
        td_est_optimizer.step()
        # Update Actor Network
        a_loss = actor_loss(model, next_states, improved_next_actions)
        a_loss.backward()
        actor_optimizer.step()

        if (training_step + 1) % 20 == 0:
            print(f"Training Step: {training_step + 1}, actor loss: {a_loss.item()}, td_est loss: {td_est_loss.item()}, average return: {average_returns[training_step-1]}")
        
        if (training_step + 1) % 1000 == 0:
            eval_return = evaluate_policy(model, eval_env, episodes=EVAL_EPISODES)
            offline_episode_returns.append(eval_return)
            print(f"Evaluation over {EVAL_EPISODES} episodes at step {training_step + 1}: Average Return: {eval_return}")

# Save Average Returns to a file
np.save(f"/home/bavish/scratch/average_returns_{ENV_NAME}_{POLICY_TYPE}.npy", average_returns)
np.save(f"/home/bavish/scratch/episode_returns_{ENV_NAME}_{POLICY_TYPE}.npy", np.array(episode_returns))
np.save(f"/home/bavish/scratch/offline_episode_returns_{ENV_NAME}_{POLICY_TYPE}.npy", np.array(offline_episode_returns))
    




