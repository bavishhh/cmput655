'''
This module defines the ActorExpert class, which combines the functionalities of an actor and an expert
ActorExpert is defined very similar to Actorexpert, the main difference is in the training loop.
This module also combines the concept of regularized corrections to the expert update (Patterson, 2021).

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
        expert_num_layers,
        expert_hidden_layer_sizes,
    ):
        super(BaseActorExpert, self).__init__()
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._actor_num_layers = actor_num_layers
        self._actor_hidden_layer_sizes = actor_hidden_layer_sizes
        self._expert_num_layers = expert_num_layers
        self._expert_hidden_layer_sizes = expert_hidden_layer_sizes

        assert len(self._actor_hidden_layer_sizes) == self._actor_num_layers, "Provided Actor Architecture is invalid! hidden_layer_sizes must match num_layers"
        assert len(self._expert_hidden_layer_sizes) == self._expert_num_layers, "Provided Expert Architecture is invalid! hidden_layer_sizes must match num_layers"

    def forward(self, state, action=None):
        pass
    
    def get_actor_params(self):
        pass

    def get_policy_params(self):
        pass

    def get_expert_params(self):
        pass

# Define the ActorExpert class for multiple policy parameterizations
class GaussianActorExpert(BaseActorExpert):
    def __init__(
        self,
        state_dim,
        action_dim,
        actor_num_layers,
        actor_hidden_layer_sizes,
        expert_num_layers,
        expert_hidden_layer_sizes,
        log_std_min=-20,
        log_std_max=2,
    ):
        super(GaussianActorExpert, self).__init__(
            state_dim,
            action_dim,
            actor_num_layers,
            actor_hidden_layer_sizes,
            expert_num_layers,
            expert_hidden_layer_sizes,
        )
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max

        # Actor Mean and Actor Std Networks (Shared)
        self._shared_actor = nn.Sequential()
        input_size = self._state_dim
        for i in range(self._actor_num_layers):
            self._shared_actor.add_module(f"actor_fc{i+1}", nn.Linear(input_size, self._actor_hidden_layer_sizes[i]))
            self._shared_actor.add_module(f"actor_relu{i+1}", nn.ReLU())
            input_size = self._actor_hidden_layer_sizes[i]
        self._actor_mean = nn.Linear(input_size, self._action_dim)
        self._actor_log_std = nn.Linear(input_size, self._action_dim)

        # Proposal Policy Mean and Proposal Policy Std Networks (Shared)
        self._shared_policy = nn.Sequential()
        input_size = self._state_dim
        for i in range(self._actor_num_layers):
            self._shared_policy.add_module(f"actor_fc{i+1}", nn.Linear(input_size, self._actor_hidden_layer_sizes[i]))
            self._shared_policy.add_module(f"actor_relu{i+1}", nn.ReLU())
            input_size = self._actor_hidden_layer_sizes[i]
        self._policy_mean = nn.Linear(input_size, self._action_dim)
        self._policy_log_std = nn.Linear(input_size, self._action_dim)

        # Expert Network and H Network (Shared)
        self._shared_expert = nn.Sequential()
        input_size = self._state_dim + self._action_dim
        for i in range(self._expert_num_layers):
            self._shared_expert.add_module(f"expert_fc{i+1}", nn.Linear(input_size, self._expert_hidden_layer_sizes[i]))
            self._shared_expert.add_module(f"expert_relu{i+1}", nn.ReLU())
            input_size = self._expert_hidden_layer_sizes[i]
        self._expert = nn.Linear(input_size, 1)
        self._td_est = nn.Linear(input_size, 1, bias=False)

    def forward(self, state, action=None, td_est=False, propose_action=False):
        # Actor and Proposal Policy forward pass
        shared_actor_out = self._shared_actor(state)
        shared_policy_out = self._shared_policy(state)
        actor_mean, actor_log_std = self._actor_mean(shared_actor_out), self._actor_log_std(shared_actor_out)
        policy_mean, policy_log_std = self._policy_mean(shared_policy_out), self._policy_log_std(shared_policy_out)
        actor_log_std = torch.clamp(actor_log_std, self._log_std_min, self._log_std_max)
        policy_log_std = torch.clamp(policy_log_std, self._log_std_min, self._log_std_max)
        actor_std = actor_log_std.exp()
        policy_std = policy_log_std.exp()

        if action is None:
            # Either propose or act
            if propose_action is True:
                state_action = torch.cat((state, policy_mean), dim=-1)
                shared_expert_out = self._shared_expert(state_action)
                q_value, h_value = self._expert(shared_expert_out), self._td_est(shared_expert_out)
                if td_est is True:
                    return (policy_mean, policy_std), q_value, h_value
                else:
                    return (policy_mean, policy_std), q_value
            else:
                state_action = torch.cat((state, actor_mean), dim=-1)
                shared_expert_out = self._shared_expert(state_action)
                q_value, h_value = self._expert(shared_expert_out), self._td_est(shared_expert_out)
                if td_est is True:
                    return (actor_mean, actor_std), q_value, h_value
                else:
                    return (actor_mean, actor_std), q_value
        else:
            # Calculate Q-value and H-value
            state_action = torch.cat((state, action), dim=-1)
            shared_expert_out = self._shared_expert(state_action)
            q_value, h_value = self._expert(shared_expert_out), self._td_est(shared_expert_out)
            if td_est is True:
                return (actor_mean, actor_std), q_value, h_value
            else:
                return (actor_mean, actor_std), q_value
    
    def get_actor_params(self):
        return self._shared_actor.parameters(), self._actor_mean.parameters(), self._actor_log_std.parameters()
    
    def get_policy_params(self):
        return self._shared_policy.parameters(), self._policy_mean.parameters(), self._policy_log_std.parameters()
    
    def get_expert_params(self):
        return self._shared_expert.parameters(), self._expert.parameters(), self._td_est.parameters()
    
class UniformActorExpert(BaseActorExpert):
    def __init__(
        self,
        state_dim,
        action_dim,
        actor_num_layers,
        actor_hidden_layer_sizes,
        expert_num_layers,
        expert_hidden_layer_sizes,
    ):
        super(UniformActorExpert, self).__init__(
            state_dim,
            action_dim,
            actor_num_layers,
            actor_hidden_layer_sizes,
            expert_num_layers,
            expert_hidden_layer_sizes,
        )

        # Actor Mean and Std Networks (Shared)
        self._shared_actor = nn.Sequential()
        input_size = self._state_dim
        for i in range(self._actor_num_layers):
            self._shared_actor.add_module(f"actor_fc{i+1}", nn.Linear(input_size, self._actor_hidden_layer_sizes[i]))
            self._shared_actor.add_module(f"actor_relu{i+1}", nn.ReLU())
            input_size = self._actor_hidden_layer_sizes[i]
        self._actor_mean = nn.Linear(input_size, self._action_dim)
        self._actor_half_range = nn.Linear(input_size, self._action_dim)

        # Proposal Policy Mean and Std Networks (Shared)
        self._shared_policy = nn.Sequential()
        input_size = self._state_dim
        for i in range(self._actor_num_layers):
            self._shared_policy.add_module(f"actor_fc{i+1}", nn.Linear(input_size, self._actor_hidden_layer_sizes[i]))
            self._shared_policy.add_module(f"actor_relu{i+1}", nn.ReLU())
            input_size = self._actor_hidden_layer_sizes[i]
        self._policy_mean = nn.Linear(input_size, self._action_dim)
        self._policy_half_range = nn.ReLU(nn.Linear(input_size, self._action_dim))

        # Expert Network (Shared)
        self._shared_expert = nn.Sequential()
        input_size = self._state_dim + self._action_dim
        for i in range(self._expert_num_layers):
            self._shared_expert.add_module(f"expert_fc{i+1}", nn.Linear(input_size, self._expert_hidden_layer_sizes[i]))
            self._shared_expert.add_module(f"expert_relu{i+1}", nn.ReLU())
            input_size = self._expert_hidden_layer_sizes[i]
        self._expert = nn.Linear(input_size, 1)
        self._td_est = nn.Linear(input_size, 1, bias=False)

    def forward(self, state, action=None, td_est=False):
        # Actor forward pass
        shared_actor_out = self._shared_actor(state)
        shared_policy_out = self._shared_policy(state)
        actor_mean, actor_half_range = self._actor_mean(shared_actor_out), self._actor_half_range(shared_actor_out)
        policy_mean, policy_half_range = self._policy_mean(shared_policy_out), self._policy_half_range(shared_policy_out)

        if action is None:
            # Either propose or act
            if propose_action is True:
                state_action = torch.cat((state, policy_mean), dim=-1)
                shared_expert_out = self._shared_expert(state_action)
                q_value, h_value = self._expert(shared_expert_out), self._td_est(shared_expert_out)
                if td_est is True:
                    return (policy_mean, policy_half_range), q_value, h_value
                else:
                    return (policy_mean, policy_half_range), q_value
            else:
                state_action = torch.cat((state, actor_mean), dim=-1)
                shared_expert_out = self._shared_expert(state_action)
                q_value, h_value = self._expert(shared_expert_out), self._td_est(shared_expert_out)
                if td_est is True:
                    return (actor_mean, actor_half_range), q_value, h_value
                else:
                    return (actor_mean, actor_half_range), q_value
        else:
            state_action = torch.cat((state, action), dim=-1)
            shared_expert_out = self._shared_expert(state_action)
            q_value, h_value = self._expert(shared_expert_out), self._td_est(shared_expert_out)
            if td_est is True:
                return (actor_mean, actor_half_range), q_value, h_value
            else:
                return (actor_mean, actor_half_range), q_value
    
    def get_actor_params(self):
        return self._shared_actor.parameters(), self._actor_mean.parameters(), self._actor_log_std.parameters()
    
    def get_policy_params(self):
        return self._shared_policy.parameters(), self._policy_mean.parameters(), self._policy_log_std.parameters()
    
    def get_expert_params(self):
        return self._shared_expert.parameters(), self._expert.parameters(), self._td_est.parameters()

class LaplacianActorExpert(BaseActorExpert):
    def __init__(
        self,
        state_dim,
        action_dim,
        actor_num_layers,
        actor_hidden_layer_sizes,
        expert_num_layers,
        expert_num_hidden_sizes,
        log_std_min,
        log_std_max
    ):
        super(LaplacianActorExpert, self).__init__(
            state_dim,
            action_dim,
            actor_num_layers,
            actor_hidden_layer_sizes,
            expert_num_layers,
            expert_num_hidden_sizes
        )

        self._log_std_min = log_std_min
        self._log_std_max = log_std_max

        # Actor Mean and Std Networks (Shared)
        self._shared_actor = nn.Sequential()
        input_size = self._state_dim
        for i in range(self._actor_num_layers):
            self._shared_actor.add_module(f"actor_fc{i+1}", nn.Linear(input_size, self._actor_hidden_layer_sizes[i]))
            self._shared_actor.add_module(f"actor_relu{i+1}", nn.ReLU())
            input_size = self._actor_hidden_layer_sizes[i]
        self._actor_mean = nn.Linear(input_size, self._action_dim)
        self._actor_log_std = nn.Linear(input_size, self._action_dim)

        # Proposal Policy Mean and Std Networks (Shared)
        self._shared_policy = nn.Sequential()
        input_size = self._state_dim
        for i in range(self._actor_num_layers):
            self._shared_policy.add_module(f"actor_fc{i+1}", nn.Linear(input_size, self._actor_hidden_layer_sizes[i]))
            self._shared_policy.add_module(f"actor_relu{i+1}", nn.ReLU())
            input_size = self._actor_hidden_layer_sizes[i]
        self._policy_mean = nn.Linear(input_size, self._action_dim)
        self._policy_log_std = nn.Linear(input_size, self._action_dim)

        # Expert Network 
        self._shared_expert = nn.Sequential()
        input_size = self._state_dim + self._action_dim
        for i in range(self._expert_num_layers):
            self._shared_expert.add_module(f"expert_fc{i+1}", nn.Linear(input_size, self._expert_hidden_layer_sizes[i]))
            self._shared_expert.add_module(f"expert_relu{i+1}", nn.ReLU())
            input_size = self._expert_hidden_layer_sizes[i]
        self._expert = nn.Linear(input_size, 1)
        self._td_est = nn.Linear(input_size, 1, bias=False)
    
    def forward(self, state, action=None, td_est=False):
        # Actor and Proposal Policy forward pass
        shared_actor_out = self._shared_actor(state)
        shared_policy_out = self._shared_policy(state)
        actor_mean, actor_log_std = self._actor_mean(shared_actor_out), self._actor_log_std(shared_actor_out)
        policy_mean, policy_log_std = self._policy_mean(shared_policy_out), self._policy_log_std(shared_policy_out)
        actor_log_std = torch.clamp(actor_log_std, self._log_std_min, self._log_std_max)
        policy_log_std = torch.clamp(policy_log_std, self._log_std_min, self._log_std_max)
        actor_std = actor_log_std.exp()
        policy_std = policy_log_std.exp()

        if action is None:
            # Either propose or act
            if propose_action is True:
                state_action = torch.cat((state, policy_mean), dim=-1)
                shared_expert_out = self._shared_expert(state_action)
                q_value, h_value = self._expert(shared_expert_out), self._td_est(shared_expert_out)
                if td_est is True:
                    return (policy_mean, policy_std), q_value, h_value
                else:
                    return (policy_mean, policy_std), q_value
            else:
                state_action = torch.cat((state, actor_mean), dim=-1)
                shared_expert_out = self._shared_expert(state_action)
                q_value, h_value = self._expert(shared_expert_out), self._td_est(shared_expert_out)
                if td_est is True:
                    return (actor_mean, actor_std), q_value, h_value
                else:
                    return (actor_mean, actor_std), q_value
        else:
            # Calculate Q-value and H-value
            state_action = torch.cat((state, action), dim=-1)
            shared_expert_out = self._shared_expert(state_action)
            q_value, h_value = self._expert(shared_expert_out), self._td_est(shared_expert_out)
            if td_est is True:
                return (actor_mean, actor_std), q_value, h_value
            else:
                return (actor_mean, actor_std), q_value

    def get_actor_params(self):
        return self._shared_actor.parameters(), self._actor_mean.parameters(), self._actor_log_std.parameters()
    
    def get_policy_params(self):
        return self._shared_policy.parameters(), self._policy_mean.parameters(), self._policy_log_std.parameters()
    
    def get_expert_params(self):
        return self._shared_expert.parameters(), self._expert.parameters(), self._td_est.parameters()


        
