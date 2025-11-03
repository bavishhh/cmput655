'''
This module defines the ActorExpert class, which combines the functionalities of an actor and an expert
ActorExpert is defined very similar to ActorCritic, the main difference is in the training loop.
This module also combines the concept of regularized corrections to the critic update (Patterson, 2021).

(Patterson, 2021) "A generalized projected bellman error for off-policy value estimation in reinforcement learning"
'''

# Import necessary libraries
import torch
import torch.nn as nn

# Define the ActorExpert class
class BaseActorExpert(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        actor_num_layers,
        actor_hidden_layer_sizes,
        critic_num_layers,
        critic_hidden_layer_sizes,
    ):
        super(BaseActorExpert, self).__init__()
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._actor_num_layers = actor_num_layers
        self._actor_hidden_layer_sizes = actor_hidden_layer_sizes
        self._critic_num_layers = critic_num_layers
        self._critic_hidden_layer_sizes = critic_hidden_layer_sizes

        assert len(self._actor_hidden_layer_sizes) == self._actor_num_layers, "Provided Actor Architecture is invalid! hidden_layer_sizes must match num_layers"
        assert len(self._critic_hidden_layer_sizes) == self._critic_num_layers, "Provided Critic Architecture is invalid! hidden_layer_sizes must match num_layers"

    def forward(self, state, action=None):
        pass
    
    def get_actor_params(self):
        pass

    def get_critic_params(self):
        pass
    
    def set_actor_params(self, params):
        pass

    def set_critic_params(self, params):
        pass

# Define the ActorExpert class for multiple policy parameterizations
class GaussianActorExpert(BaseActorExpert):
    def __init__(
        self,
        state_dim,
        action_dim,
        actor_num_layers,
        actor_hidden_layer_sizes,
        critic_num_layers,
        critic_hidden_layer_sizes,
        log_std_min=-20,
        log_std_max=2,
    ):
        super(GaussianActorExpert, self).__init__(
            state_dim,
            action_dim,
            actor_num_layers,
            actor_hidden_layer_sizes,
            critic_num_layers,
            critic_hidden_layer_sizes,
        )
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max

        # Mean and Std Networks (Shared)
        self._shared_policy = nn.Sequential()
        input_size = self._state_dim
        for i in range(self._actor_num_layers):
            self._shared_policy.add_module(f"actor_fc{i+1}", nn.Linear(input_size, self._actor_hidden_layer_sizes[i]))
            self._shared_policy.add_module(f"actor_relu{i+1}", nn.ReLU())
            input_size = self._actor_hidden_layer_sizes[i]
        self._mean = nn.Linear(input_size, self._action_dim)
        self._log_std = nn.Linear(input_size, self._action_dim)

        # Critic Network 
        self._shared_critic = nn.Sequential()
        input_size = self._state_dim + self._action_dim
        for i in range(self._critic_num_layers):
            self._shared_critic.add_module(f"critic_fc{i+1}", nn.Linear(input_size, self._critic_hidden_layer_sizes[i]))
            self._shared_critic.add_module(f"critic_relu{i+1}", nn.ReLU())
            input_size = self._critic_hidden_layer_sizes[i]
        self._critic = nn.Linear(input_size, 1)
        self._td_est = nn.Linear(input_size, 1)

    def forward(self, state, action=None, td_est=False):
        # Actor forward pass
        shared_out = self._shared_policy(state)
        mean = self._mean(shared_out)
        log_std = self._log_std(shared_out)
        log_std = torch.clamp(log_std, self._log_std_min, self._log_std_max)
        std = log_std.exp()

        if action is not None:
            # Critic forward pass
            state_action = torch.cat([state, action], dim=-1)
            q_value = self._critic(self._shared_critic(state_action))
            return (mean, std), q_value if td_est == False else self._td_est(self._shared_critic(state_action))
        else:
            state_action = torch.cat([state, mean], dim=-1)
            return (mean, std), self._critic(self._shared_critic(state_action)) if td_est == False else self._td_est(self._shared_critic(state_action))
    
    def get_actor_params(self):
        return self._shared_policy.parameters(), self._mean.parameters(), self._log_std.parameters()
    
    def get_critic_params(self):
        return self._shared_critic.parameters(), self._critic.parameters(), self._td_est.parameters()

    def set_actor_params(self, params):
        assert len(params) == 3, "Invalid number of parameter sets for actor!"
        shared_policy_params, mean_params, log_std_params = params
        self._shared_policy.load_state_dict(shared_policy_params)
        self._mean.load_state_dict(mean_params)
        self._log_std.load_state_dict(log_std_params)
    
    def set_critic_params(self, params):
        assert len(params) == 3, "Invalid number of parameter sets for critic!"
        shared_critic_params, critic_params, td_est_params = params
        self._shared_critic.load_state_dict(shared_critic_params)
        self._critic.load_state_dict(critic_params)
        self._td_est.load_state_dict(td_est_params)
    
class UniformActorExpert(BaseActorExpert):
    def __init__(
        self,
        state_dim,
        action_dim,
        actor_num_layers,
        actor_hidden_layer_sizes,
        critic_num_layers,
        critic_hidden_layer_sizes,
    ):
        super(UniformActorExpert, self).__init__(
            state_dim,
            action_dim,
            actor_num_layers,
            actor_hidden_layer_sizes,
            critic_num_layers,
            critic_hidden_layer_sizes,
        )

        # Mean and Std Networks (Shared)
        self._shared_policy = nn.Sequential()
        input_size = self._state_dim
        for i in range(self._actor_num_layers):
            self._shared_policy.add_module(f"actor_fc{i+1}", nn.Linear(input_size, self._actor_hidden_layer_sizes[i]))
            self._shared_policy.add_module(f"actor_relu{i+1}", nn.ReLU())
            input_size = self._actor_hidden_layer_sizes[i]
        self._mean = nn.Linear(input_size, self._action_dim)
        self._half_range = nn.ReLU(nn.Linear(input_size, self._action_dim))

        # Critic Network 
        self._shared_critic = nn.Sequential()
        input_size = self._state_dim + self._action_dim
        for i in range(self._critic_num_layers):
            self._shared_critic.add_module(f"critic_fc{i+1}", nn.Linear(input_size, self._critic_hidden_layer_sizes[i]))
            self._shared_critic.add_module(f"critic_relu{i+1}", nn.ReLU())
            input_size = self._critic_hidden_layer_sizes[i]
        self._critic = nn.Linear(input_size, 1)
        self._td_est = nn.Linear(input_size, 1)

    def forward(self, state, action=None):
        # Actor forward pass
        shared_out = self._shared_policy(state)
        mean = self._mean(shared_out)
        half_range = self._half_range(shared_out)

        if action is not None:
            # Critic forward pass
            state_action = torch.cat([state, action], dim=-1)
            q_value = self._critic(self._shared_critic(state_action))
            return (mean, half_range), q_value
        else:
            state_action = torch.cat([state, mean], dim=-1)
            return (mean, half_range), self._critic(self._shared_critic(state_action)) 
    
    def get_actor_params(self):
        return self._shared_policy.parameters(), self._mean.parameters(), self._half_range.parameters()
    
    def get_critic_params(self):
        return self._shared_critic.parameters(), self._critic.parameters(), self._td_est.parameters()

    def set_actor_params(self, params):
        assert len(params) == 3, "Invalid number of parameter sets for actor!"
        shared_policy_params, mean_params, half_range_params = params
        self._shared_policy.load_state_dict(shared_policy_params)
        self._mean.load_state_dict(mean_params)
        self._half_range.load_state_dict(half_range_params)
    
    def set_critic_params(self, params):
        assert len(params) == 3, "Invalid number of parameter sets for critic!"
        shared_critic_params, critic_params, td_est_params = params
        self._shared_critic.load_state_dict(shared_critic_params)
        self._critic.load_state_dict(critic_params)
        self._td_est.load_state_dict(td_est_params)

class LaplacianActorExpert(BaseActorExpert):
    def __init__(
        self,
        state_dim,
        action_dim,
        actor_num_layers,
        actor_hidden_layer_sizes,
        critic_num_layers,
        critic_num_hidden_sizes,
        log_std_min,
        log_std_max
    ):
        super(LaplacianActorExpert, self).__init__(
            state_dim,
            action_dim,
            actor_num_layers,
            actor_hidden_layer_sizes,
            critic_num_layers,
            critic_num_hidden_sizes
        )

        self._log_std_min = log_std_min
        self._log_std_max = log_std_max

        # Mean and Std Networks (Shared)
        self._shared_policy = nn.Sequential()
        input_size = self._state_dim
        for i in range(self._actor_num_layers):
            self._shared_policy.add_module(f"actor_fc{i+1}", nn.Linear(input_size, self._actor_hidden_layer_sizes[i]))
            self._shared_policy.add_module(f"actor_relu{i+1}", nn.ReLU())
            input_size = self._actor_hidden_layer_sizes[i]
        self._mean = nn.Linear(input_size, self._action_dim)
        self._log_std = nn.Linear(input_size, self._action_dim)

        # Critic Network 
        self._shared_critic = nn.Sequential()
        input_size = self._state_dim + self._action_dim
        for i in range(self._critic_num_layers):
            self._shared_critic.add_module(f"critic_fc{i+1}", nn.Linear(input_size, self._critic_hidden_layer_sizes[i]))
            self._shared_critic.add_module(f"critic_relu{i+1}", nn.ReLU())
            input_size = self._critic_hidden_layer_sizes[i]
        self._critic = nn.Linear(input_size, 1)
        self._td_est = nn.Linear(input_size, 1)
    
    def forward(self, state, action=None):
        # Actor forward pass
        shared_out = self._shared_policy(state)
        mean = self._mean(shared_out)
        log_std = self._log_std(shared_out)
        log_std = torch.clamp(log_std, self._log_std_min, self._log_std_max)
        std = log_std.exp()

        if action is not None:
            # Critic forward pass
            state_action = torch.cat([state, action], dim=-1)
            q_value = self._critic(self._shared_critic(state_action))
            return (mean, std), q_value
        else:
            state_action = torch.cat([state, mean], dim=-1)
            return (mean, std), self._critic(self._shared_critic(state_action))
    
    def get_actor_params(self):
        return self._shared_policy.parameters(), self._mean.parameters(), self._log_std.parameters()
    
    def get_critic_params(self):
        return self._shared_critic.parameters(), self._critic.parameters(), self._td_est.parameters()

    def set_actor_params(self, params):
        assert len(params) == 3, "Invalid number of parameter sets for actor!"
        shared_policy_params, mean_params, log_std_params = params
        self._shared_policy.load_state_dict(shared_policy_params)
        self._mean.load_state_dict(mean_params)
        self._log_std.load_state_dict(log_std_params)
    
    def set_critic_params(self, params):
        assert len(params) == 3, "Invalid number of parameter sets for critic!"
        shared_critic_params, critic_params, td_est_params = params
        self._shared_critic.load_state_dict(shared_critic_params)
        self._critic.load_state_dict(critic_params)
        self._td_est.load_state_dict(td_est_params)


        
