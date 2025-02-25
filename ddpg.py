import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import random


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size)

    def add(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        return torch.cat(states), torch.cat(actions), torch.cat(next_states), torch.tensor(rewards, device=device), torch.tensor(dones, device=device)

    def __len__(self):
        return len(self.memory)


# Policy Network (Actor)
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers=4):
        super(PolicyNetwork, self).__init__()
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.model = nn.Sequential(*layers)
        
        # Initialize weights with small values for better convergence
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.01)
            module.bias.data.fill_(0.01)
    
    def forward(self, state):
        # For least squares optimization, we don't need to apply tanh
        # as we're not limited to a specific action range
        return self.model(state)


# Q-Network (Critic)
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers=4):
        super(QNetwork, self).__init__()
        self.state_linear = nn.Linear(state_dim, hidden_dim)
        
        layers = [nn.Linear(hidden_dim + action_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.model = nn.Sequential(*layers)
        
        # Initialize weights for better convergence
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.01)
            module.bias.data.fill_(0.01)
    
    def forward(self, state, action):
        x = torch.relu(self.state_linear(state))
        x = torch.cat([x, action], dim=1)
        return self.model(x)


# Ornstein-Uhlenbeck Noise for exploration
class OUNoise:
    def __init__(self, size, mu=0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
        
    def reset(self):
        self.state = np.copy(self.mu)
        
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state, dtype=torch.float32, device=device)


# DDPG Agent
class DDPG:
    def __init__(self, state_dim, action_dim, hidden_dim, policy_lr=1e-4, q_lr=1e-3, 
                 batch_size=64, buffer_size=10000, tau=0.005, gamma=0.99, noise_decay=0.99):
        # Initialize networks
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_target = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q_net_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=q_lr)
        
        # Replay Buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Hyperparameters
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        
        # Exploration noise
        self.noise = OUNoise(action_dim)
        self.noise_scale = 1.0
        self.noise_decay = noise_decay
        
        # Initialize target networks
        self.hard_update(self.policy_target, self.policy)
        self.hard_update(self.q_net_target, self.q_net)
        
        # For tracking learning progress
        self.q_losses = []
        self.policy_losses = []
        self.rewards = []
    
    def select_action(self, state, add_noise=True):
        """Select action according to policy with optional noise for exploration"""
        self.policy.eval()
        with torch.no_grad():
            action = self.policy(state)
        self.policy.train()
        
        if add_noise:
            noise = self.noise_scale * self.noise.sample()
            action = action + noise
            
        return action
    
    def store_transition(self, state, action, next_state, reward, done):
        """Store transition in replay buffer"""
        self.memory.add(state, action, next_state, reward, done)
    
    def learn(self):
        """Update policy and Q networks using batch from replay buffer"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample mini-batch from replay buffer
        states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size)
        
        # Update critic (Q-network)
        with torch.no_grad():
            next_actions = self.policy_target(next_states)
            next_q_values = self.q_net_target(next_states, next_actions).squeeze()
            target_q_values = rewards + (1 - dones.float()) * self.gamma * next_q_values
        
        current_q_values = self.q_net(states, actions).squeeze()
        q_loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.q_optimizer.step()
        
        # Update actor (policy network)
        policy_loss = -self.q_net(states, self.policy(states)).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # Soft update of target networks
        self.soft_update(self.q_net_target, self.q_net)
        self.soft_update(self.policy_target, self.policy)
        
        # Decay noise
        self.noise_scale *= self.noise_decay
        
        # Track losses
        self.q_losses.append(q_loss.item())
        self.policy_losses.append(policy_loss.item())
        
        return q_loss.item(), policy_loss.item()
    
    def soft_update(self, target, source):
        """Soft update model parameters: θ_target = τ*θ_source + (1-τ)*θ_target"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def hard_update(self, target, source):
        """Hard update model parameters: θ_target = θ_source"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


class LeastSquaresEnvironment(gym.Env):
    """Environment for solving least squares problems using RL"""
    
    def __init__(self, dimension=10, max_steps=100):
        super(LeastSquaresEnvironment, self).__init__()
        
        self.dimension = dimension
        self.max_steps = max_steps
        
        # Define action and observation spaces
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(dimension,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(dimension,), dtype=np.float32)
        
        # Create a random least squares problem
        self.reset_problem()
        
        # Track current state and step count
        self.current_step = 0
        self.best_solution = None
        self.best_loss = float('inf')
        self.optimal_solution = None
        self.initial_loss = None
        
    def reset_problem(self):
        """Generate a new random least squares problem"""
        self.A = torch.randn(self.dimension, self.dimension, device=device)
        self.b = torch.randn(self.dimension, device=device)
        
        # Calculate the optimal solution for evaluation
        self.optimal_solution = torch.linalg.lstsq(self.A, self.b.unsqueeze(1)).solution.squeeze()
        self.optimal_loss = torch.norm(self.A @ self.optimal_solution - self.b, p=2).item()
        
    def reset(self, seed=None, options=None):
        """Reset environment state"""
        super().reset(seed=seed)
        
        # Generate new problem with some probability
        if np.random.random() < 0.1:
            self.reset_problem()
            
        # Start with random solution
        self.state = torch.randn(self.dimension, device=device) * 0.1
        self.current_step = 0
        self.best_solution = self.state.clone()
        self.best_loss = self.compute_loss(self.state)
        self.initial_loss = self.best_loss
        
        return self.state.unsqueeze(0), {}
    
    def compute_loss(self, x):
        """Compute least squares loss ||Ax - b||^2"""
        return torch.norm(self.A @ x - self.b, p=2).item()
    
    def compute_gradient(self, x):
        """Compute gradient of loss function"""
        return 2 * torch.t(self.A) @ (self.A @ x - self.b)
    
    def step(self, action):
        """Take a step in the environment using the provided action"""
        action = action.squeeze()
        
        # In our case, the action directly updates the state (solution vector)
        next_state = action
        self.state = next_state
        
        # Compute loss and reward
        loss = self.compute_loss(self.state)
        
        # Reward is negative loss (we want to maximize reward by minimizing loss)
        # We scale the reward to avoid very large negative values
        reward = -loss
        
        # Update best solution if current is better
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_solution = self.state.clone()
        
        # Check if we've converged or reached max steps
        self.current_step += 1
        done = loss < 1e-6 or self.current_step >= self.max_steps
        
        # Additional info
        info = {
            'loss': loss,
            'optimal_loss': self.optimal_loss,
            'relative_error': abs(loss - self.optimal_loss) / (self.optimal_loss + 1e-10),
            'improvement': (self.initial_loss - loss) / self.initial_loss if self.initial_loss > 0 else 0,
            'step': self.current_step,
            'gradient_norm': torch.norm(self.compute_gradient(self.state)).item()
        }
        
        return self.state.unsqueeze(0), reward, done, False, info


def train_ddpg_for_least_squares(episodes=1000, max_steps=100, eval_interval=10):
    """Train a DDPG agent to solve least squares problems"""
    
    # Create environment and agent
    dimension = 10
    env = LeastSquaresEnvironment(dimension=dimension, max_steps=max_steps)
    
    agent = DDPG(
        state_dim=dimension,
        action_dim=dimension,
        hidden_dim=128,
        policy_lr=1e-4,
        q_lr=1e-3,
        batch_size=64,
        buffer_size=10000,
        tau=0.005,
        gamma=0.99
    )
    
    # Track results
    episode_rewards = []
    final_losses = []
    improvements = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state, add_noise=(episode < episodes * 0.8))
            
            # Take step in environment
            next_state, reward, done, _, info = env.step(action)
            
            # Store transition and learn
            agent.store_transition(state, action, next_state, reward, done)
            
            # Update agent
            if len(agent.memory) >= agent.batch_size:
                agent.learn()
            
            # Update state and tracking variables
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Store episode results
        episode_rewards.append(episode_reward)
        final_losses.append(info['loss'])
        improvements.append(info['improvement'])
        
        # Print progress
        if (episode + 1) % eval_interval == 0:
            print(f"Episode {episode+1}/{episodes}")
            print(f"  Final Loss: {info['loss']:.6f}, Optimal Loss: {env.optimal_loss:.6f}")
            print(f"  Improvement: {info['improvement']:.2%}")
            print(f"  Gradient Norm: {info['gradient_norm']:.6f}")
            print(f"  Steps: {info['step']}")
            print("-" * 50)
    
    # Evaluate the final policy without exploration noise
    evaluate_agent(agent, env, num_episodes=10)
    
    # Plot results
    plot_results(episode_rewards, final_losses, improvements)
    
    return agent, env


def evaluate_agent(agent, env, num_episodes=10):
    """Evaluate agent performance without exploration noise"""
    total_reward = 0
    total_loss = 0
    total_improvement = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(env.max_steps):
            # Select action without noise
            action = agent.select_action(state, add_noise=False)
            
            # Take step in environment
            next_state, reward, done, _, info = env.step(action)
            
            # Update state and tracking variables
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        total_reward += episode_reward
        total_loss += info['loss']
        total_improvement += info['improvement']
    
    avg_reward = total_reward / num_episodes
    avg_loss = total_loss / num_episodes
    avg_improvement = total_improvement / num_episodes
    
    print("\nEvaluation Results:")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Average Loss: {avg_loss:.6f}")
    print(f"  Average Improvement: {avg_improvement:.2%}")
    
    return avg_reward, avg_loss, avg_improvement


def plot_results(rewards, losses, improvements):
    """Plot training progress"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 3, 2)
    plt.plot(losses)
    plt.title('Final Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.yscale('log')
    
    plt.subplot(1, 3, 3)
    plt.plot(improvements)
    plt.title('Improvement Ratio')
    plt.xlabel('Episode')
    plt.ylabel('Improvement')
    
    plt.tight_layout()
    plt.show()


def compare_methods(dimension=10, random_seed=42):
    """Compare DDPG to classic least squares solution"""
    # Set random seeds
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Create a random problem
    A = torch.randn(dimension, dimension, device=device)
    b = torch.randn(dimension, device=device)
    
    # Optimal solution using direct method
    optimal_solution = torch.linalg.lstsq(A, b.unsqueeze(1)).solution.squeeze()
    optimal_loss = torch.norm(A @ optimal_solution - b, p=2).item()
    
    print("Direct Least Squares Solution:")
    print(f"  Loss: {optimal_loss:.6f}")
    
    # Create environment with the same problem
    env = LeastSquaresEnvironment(dimension=dimension)
    env.A = A
    env.b = b
    env.optimal_solution = optimal_solution
    env.optimal_loss = optimal_loss
    
    # Train DDPG agent
    agent, _ = train_ddpg_for_least_squares(episodes=200, max_steps=50, eval_interval=20)
    
    # Final evaluation
    _, avg_loss, _ = evaluate_agent(agent, env, num_episodes=10)
    
    print("\nComparison:")
    print(f"  Direct Method Loss: {optimal_loss:.6f}")
    print(f"  DDPG Method Loss: {avg_loss:.6f}")
    print(f"  Relative Difference: {abs(avg_loss - optimal_loss) / optimal_loss:.2%}")


if __name__ == "__main__":
    # Train the agent
    agent, env = train_ddpg_for_least_squares(episodes=500, max_steps=100)
    
    # Compare with direct method
    compare_methods(dimension=10)
