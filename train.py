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
parser.add_argument("--env_name", type=str, default="MountainCarContinuous-v0", help="Name of the Gym environment")
parser.add_argument("--actor_num_layers", type=int, default=2, help="Number of layers in the actor network")
parser.add_argument("--expert_num_layers", type=int, default=2, help="Number of layers in the critic network")
parser.add_argument("--actor_hidden_layer_sizes", type=int, nargs='+', default=[128, 128], help="Sizes of hidden layers in the actor network")
parser.add_argument("--expert_hidden_layer_sizes", type=int, nargs='+', default=[128, 128], help="Sizes of hidden layers in the critic network")
parser.add_argument("--actor_lr", type=float, default=1e-4, help="Learning rate for the Actor optimizer")
parser.add_argument("--expert_lr", type=float, default=1e-3, help="Learning rate for Critic optimizer")
parser.add_argument("--max_lr", type=float, default=1e-3, help="Learning rate for Q-value maximisation GD")
parser.add_argument("--max_timesteps", type=int, default=200, help="Maximum timesteps per episode")
parser.add_argument("--training_steps", type=int, default=int(1e5), help="Number of total training steps")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--policy_type", type=str, default="gaussian", choices=["gaussian", "uniform", "laplacian"], help="Type of policy parameterization")
parser.add_argument("--alpha", type=float, default=0.1, help="Entropy Regularization coefficient for proposal policy")
parser.add_argument("--beta", type=float, default=0.1, help="Regularization coefficient for critic updates")
parser.add_argument("--tau", type=int, default=10, help="Number of gradient steps for Q-value maximization")
args = parser.parse_args()

# Define the device to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters - Given
ENV_NAME = args.env_name
ACTOR_NUM_LAYERS = args.actor_num_layers
EXPERT_NUM_LAYERS = args.expert_num_layers
ACTOR_HIDDEN_LAYER_SIZES = args.actor_hidden_layer_sizes
EXPERT_HIDDEN_LAYER_SIZES = args.expert_hidden_layer_sizes
ACTOR_LR = args.actor_lr
EXPERT_LR = args.expert_lr
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
INIT_EXPLORE_FRACTION = 0.01  
PRINT_FREQUENCY = 100
EVAL_FREQUENCY = 1000
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
    expert_num_layers=EXPERT_NUM_LAYERS,
    expert_hidden_layer_sizes=EXPERT_HIDDEN_LAYER_SIZES,
).to(device)

shared_policy, mean_policy, std_policy = model.get_policy_params()
shared_actor, mean_actor, std_actor = model.get_actor_params()
shared_expert, expert_head, td_est_head = model.get_expert_params()

# Define function to get distribution
def get_dist(mean, param, policy_type=POLICY_TYPE, epsilon=EPSILON):
    if policy_type == "gaussian":
        dist = torch.distributions.Normal(mean, param)
    elif policy_type == "uniform":
        lower_bound = mean - param
        upper_bound = mean + param
        dist = torch.distributions.Uniform(lower_bound - epsilon, upper_bound + epsilon)
    elif policy_type == "laplacian":
        dist = torch.distributions.Laplace(mean, param)
    return dist

# Define loss functions for each parameterization of actor and proposal policy
def actor_loss(model, next_states, improved_actions, policy_type=POLICY_TYPE):
    (actor_mean, actor_std), _ = model(next_states, td_est=False, propose_action=False)
    actor_dist = get_dist(actor_mean, actor_std)
    log_probs = actor_dist.log_prob(improved_actions).sum(-1)
    return -log_probs.mean()

def policy_loss(model, next_states, improved_actions, policy_type=POLICY_TYPE, alpha=ALPHA):
    (policy_mean, policy_std), _ = model(next_states, td_est=False, propose_action=True)
    policy_dist = get_dist(policy_mean, policy_std)
    log_probs = policy_dist.log_prob(improved_actions).sum(-1).mean()
    entropy = policy_dist.entropy().sum(-1).mean()
    return -log_probs - alpha * entropy

# Define the optimizers [Learning rates of critic and td_est are same as design choice (Patterson, 2021)]
actor_optimizer = optim.Adam(list(shared_actor) + list(mean_actor) + list(std_actor), lr=ACTOR_LR)
policy_optimizer = optim.Adam(list(shared_policy) + list(mean_policy) + list(std_policy), lr=ACTOR_LR)
expert_optimizer = optim.Adam(list(shared_expert) + list(expert_head), lr=EXPERT_LR)
td_est_optimizer = optim.Adam(list(td_est_head), lr=EXPERT_LR)

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

# Define a function to calculate TD targets 
def compute_td_target(model, rewards, next_states, dones, gamma=GAMMA):
    with torch.no_grad():
        # Sample next actions from the current policy
        (next_mean, next_param), _ = model(next_states, td_est=False, propose_action=True)
        dist = get_dist(next_mean, next_param, POLICY_TYPE)
        next_actions = dist.rsample()
    # Perform Gradient Ascent to find maximum Q-value for next states without updating model parameters
    saved_state = {k: v.clone() for k, v in model.state_dict().items()}
    def loss_fn(next_state, next_action):
        _, q_value = model(next_state, next_action)
        return q_value
    next_actions = next_actions.detach().requires_grad_(True)
    next_states = next_states.detach().requires_grad_(True)
    def f(action):
        return loss_fn(next_states, action).squeeze(-1)
    improved_next_actions = next_actions.clone()
    for _ in range(TAU):
        # Gradient ascent step
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

# Define function to compute critic gradients
def compute_expert_gradients(model, states, actions, next_states, improved_next_actions):
    _, current_q_values = model(states, actions, td_est=False)        # [B,1]
    _, next_q_values    = model(next_states, improved_next_actions, td_est=False)
    current_q_values = current_q_values.squeeze(-1)     # [B]
    next_q_values    = next_q_values.squeeze(-1)
    params = list(model.get_expert_params()[0]) + list(model.get_expert_params()[1])
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

# Define function to compute shared critic outputs
def compute_shared_expert_outputs(model, states, actions):
    shared_expert, _, _ = model.get_expert_params()
    x = torch.cat((states, actions), dim=1)
    shared_expert = list(shared_expert)
    weights = shared_expert[::2]
    biases = shared_expert[1::2]
    for weight, bias in zip(weights, biases):
        x = F.relu(torch.matmul(x, weight.transpose(0, 1)) + bias)
    return x

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
                (mean, param) = model(state_tensor, propose_action=False)[0]
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
offline_episode_returns = []
actor_losses = []
policy_losses = []

for training_step in range(TRAINING_STEPS):
    if training_step < int(INIT_EXPLORE_FRACTION * TRAINING_STEPS):
        # Take random actions for initial exploration
        action = env.action_space.sample()
    else:
        # Sample action from the current policy
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        (mean, param) = model(state_tensor, propose_action=False)[0]
        dist = get_dist(mean, param, POLICY_TYPE)
        action = dist.rsample().cpu().detach().numpy()[0]
        action = np.clip(action, action_low, action_high)
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    replay_buffer.add((state, action, reward, next_state, done))

    # Sample a batch from the replay buffer
    states, actions, rewards, next_states, dones = replay_buffer.sample()
    states = torch.tensor(states, dtype=torch.float32, requires_grad=True).to(device)
    actions = torch.tensor(actions, dtype=torch.float32, requires_grad=True).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32, requires_grad=True).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

    # Skip the update steps if its the initial exploration phase
    if training_step >= int(INIT_EXPLORE_FRACTION * TRAINING_STEPS):
        # Stop accumulating past gradients
        expert_optimizer.zero_grad()
        td_est_optimizer.zero_grad()
        actor_optimizer.zero_grad()
        policy_optimizer.zero_grad()

        # Compute TD targets
        td_targets, next_actions, improved_next_actions = compute_td_target(model, rewards, next_states, dones)

        # Update Critic parameters
        _, q_values, td_ests = model(states, actions, td_est=True, propose_action=False)
        current_q_gradient, next_q_gradient = compute_expert_gradients(model, states, actions, next_states, improved_next_actions)
        index = 0
        for param in list(model.get_expert_params()[0]) + list(model.get_expert_params()[1]):
            cgrad, qgrad = current_q_gradient[index], next_q_gradient[index]
            if len(param.shape) > 1:
                td_ests = td_ests.unsqueeze(-1)
                td_targets = td_targets.unsqueeze(-1)
            term1 = GAMMA * td_ests * qgrad
            term2 = td_targets * cgrad
            param.grad = term1.sum(dim=0) - term2.sum(dim=0)
            if len(param.shape) > 1:
                td_ests = td_ests.squeeze(-1)
                td_targets = td_targets.squeeze(-1)
            index += 1
        expert_optimizer.step()

        # Update TD Estimate Network
        for param in list(model.get_expert_params()[-1]):
            term1 = BETA * param
            term2 = torch.matmul(compute_shared_expert_outputs(model, states, actions).transpose(0, 1), td_targets - td_ests).transpose(0, 1)
            param.grad = term1 - term2
        td_est_optimizer.step()

        # Update Actor Network
        a_loss = actor_loss(model, next_states, improved_next_actions)
        p_loss = policy_loss(model, next_states, improved_next_actions, alpha=ALPHA)
        a_loss.backward(retain_graph=True)
        p_loss.backward()
        actor_optimizer.step()
        policy_optimizer.step()

        if (training_step + 1) % PRINT_FREQUENCY == 0:
            print(f"Training Step: {training_step + 1}, actor loss: {a_loss.item()}, proposal policy loss: {p_loss.item()}")
            actor_losses.append(a_loss.item())
            policy_losses.append(p_loss.item())
        
        if (training_step + 1) % EVAL_FREQUENCY == 0:
            eval_return = evaluate_policy(model, eval_env, episodes=EVAL_EPISODES)
            offline_episode_returns.append(eval_return)
            print(f"Evaluation over {EVAL_EPISODES} episodes at step {training_step + 1}: Average Return: {eval_return}")
        
    training_step += 1

# Save Average Returns to a file
np.save(f"/home/saigp/scratch/cmput655/actor_losses_{ENV_NAME}_{POLICY_TYPE}.npy", np.array(actor_losses))
np.save(f"/home/saigp/scratch/cmput655/policy_losses_{ENV_NAME}_{POLICY_TYPE}.npy", np.array(policy_losses))
np.save(f"/home/saigp/scratch/cmput655/offline_episode_returns_{ENV_NAME}_{POLICY_TYPE}.npy", np.array(offline_episode_returns))
    




