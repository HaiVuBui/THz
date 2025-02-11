# Replay Buffer for storing experiences
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque


device = 'cuda'


class ReplayBuffer():
    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size)

    def add(self, state, action, next_state, reward):
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, next_states, rewards = zip(*batch)
        return states, actions, next_states, rewards

    def __len__(self):
        return len(self.memory)

# Policy Network (Actor)
class policy_net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(policy_net, self).__init__()
        self.ln1 = nn.Linear(state_dim, hidden_dim)
        self.ln2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.Linear(hidden_dim, action_dim)
        self.tanh = nn.Tanh()  # Output action scaled between -1 and 1

    def forward(self, state):
        x = torch.relu(self.ln1(state))
        x = torch.relu(self.ln2(x))
        x = self.tanh(self.ln3(x))  # Final activation for action
        return x

# Q-Network (Critic)
class Q_net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Q_net, self).__init__()
        self.ln1 = nn.Linear(state_dim, hidden_dim)
        self.ln2 = nn.Linear(hidden_dim + action_dim, hidden_dim)  # Action concatenated
        self.ln3 = nn.Linear(hidden_dim, 1)  # Output Q-value

    def forward(self, state, action):
        x = torch.relu(self.ln1(state))
        x = torch.relu(self.ln2(torch.cat([x, action], dim=1)))  # Concatenate state and action
        x = self.ln3(x)  # Output Q-value (no activation)
        return x

# DDPG Agent
class ddpg():
    def __init__(self, state_dim, action_dim, hidden_dim, policy_lr=1e-3, Q_lr=1e-3, batch_size=32, buffer_size=1000, tau=5e-3, gamma=0.99):
        # Initialize networks
        self.policy = policy_net(state_dim,action_dim, hidden_dim).to(device)
        self.policy_target = policy_net(state_dim, action_dim, hidden_dim).to(device)
        self.Q_net = Q_net(state_dim, action_dim, hidden_dim).to(device)
        self.Q_net_target = Q_net(state_dim, action_dim, hidden_dim).to(device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.Q_net_optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=Q_lr)
        
        # Replay Buffer
        self.memory = ReplayBuffer(buffer_size)

        #other hyperparameters
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        
        # Action noise for exploration
        self.noise = torch.tensor([[0.0]], dtype=torch.float32, device=device)
        
        # Initialize target networks to match current networks
        self._update_target(self.policy, self.policy_target, tau=1)
        self._update_target(self.Q_net, self.Q_net_target, tau=1)
    
    def act(self, state):
        action = self.policy(state).detach()
        return action  
    
    def add(self, state, action, next_state, reward):
        self.memory.add(state, action, next_state, reward)
    
    def learn(self, batch_size=32):
        batch_size = batch_size
        
        # Sample batch from replay buffer
        state, action, next_state, reward = self.memory.sample(batch_size)
        
        # Convert to torch tensors
        state = torch.cat(state)
        action = torch.cat(action)
        reward = torch.cat(reward)
        
        # Compute Q-value estimates
        state_action_values = self.Q_net(state, action).squeeze(1)
        
        # Mask for non-terminal states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state)), dtype=torch.bool, device=device)
        non_final_next_states = torch.cat([s for s in next_state if s is not None])
        
        # Compute next state values
        next_state_values = torch.zeros(batch_size, device=device)
        next_action = self.policy_target(non_final_next_states)
        
        with torch.no_grad():
            next_state_values[non_final_mask] = self.Q_net_target(non_final_next_states, next_action).squeeze(1)
        
        # Compute target Q-values
        target_state_action_values = next_state_values * self.gamma + reward
        
        # Define loss function (MSE Loss for Q-network)
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, target_state_action_values)
        
        # Update Q-network
        self.Q_net_optimizer.zero_grad()
        loss.backward()
        self.Q_net_optimizer.step()
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        J = -self.Q_net(state, self.policy(state)).mean()  # Policy loss
        J.backward()
        self.policy_optimizer.step()
        
        # Soft update target networks
        self._update_target(self.Q_net, self.Q_net_target, self.tau)
        self._update_target(self.policy, self.policy_target, self.tau)
    
    def _update_target(self, current, target, tau):
        current_dict = current.state_dict()
        target_dict = target.state_dict()
        for key in current_dict:
            target_dict[key] = current_dict[key] * tau + target_dict[key] * (1 - tau)
        target.load_state_dict(target_dict)

# Training Function

def train(agent, env, episodes=100, updates=1, steps=100):
    for i_episode in range(episodes):
        # Reset environment
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
        
        for step in range(steps):
            # Agent selects action
            action = agent.act(state=state)
            
            # Execute action in environment
            # print(state)
            # print(action)
            observation, reward, terminated, truncated, _ = env.step(action)
            reward = torch.tensor([reward], device=device)
            
            # Handle terminal state
            next_state = None if terminated else observation
            
            # Store experience in replay buffer
            agent.add(state, action, next_state, reward)
            
            # Update state
            state = next_state
            
            # Train if memory has enough samples
            if len(agent.memory) >= agent.batch_size:
                for update in range(updates):
                    agent.learn()
            
            if terminated or truncated:
                break
        print(f'episode {i_episode} finished')
        print('------------------------------')
        

