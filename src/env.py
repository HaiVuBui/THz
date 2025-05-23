import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from objective import Objective
from gymnasium.envs.registration import register

from objective import construct_V


class CustomEnv(gym.Env):  # observation and action
    """
    Custom Environment that follows gym interface.
    """

    def __init__(self, objective: Objective, stop_loss: float = 1e-5):
        super(CustomEnv, self).__init__()

        self.best_solution: float = float("inf")
        self.objective: Objective = objective
        self.stop_loss: float = stop_loss

        self.action_space = Box(
            low=-10.0,
            high=10.0,
            shape=(self.objective.action_shape(),),
            dtype=np.float32,
        )

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.objective.state_shape(),),
            dtype=np.float32,
        )

        # Environment parameters
        self.state: torch.Tensor = torch.zeros((self.observation_space.shape))
        self.steps_taken: int = 0
        self.max_steps: int = 1000  # Maximum steps per episode
        self.best_loss: float = float("inf")
        self.best_params: torch.Tensor = torch.zeros((self.observation_space.shape))

    def reset(self, seed: int = 100, options=None):
        """
        Reset the environment to an initial state and return the initial observation.
        """
        super().reset(seed=100)

        # Reset episode parameters
        self.steps_taken = 0
        self.best_loss = float("inf")
        self.best_params = None

        # Return initial observation and info
        return self.state, {}

    def step(self, action: torch.Tensor):
        """
        Take a step in the environment using the given action.
        Returns next_state, reward, terminated, truncated, info
        """
        # Ensure action is within bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # compute parameters
        M = self.objective.M
        N = self.objective.N
        K = self.objective.K
        D = self.objective.D
        F = self.objective.F
        S = self.objective.S

        W_module = torch.from_numpy(action[: N * K])
        W_polar = torch.from_numpy(action[N * K : 2 * N * K])
        W = torch.polar(W_module, W_polar).reshape(N, K)

        V_polar = torch.from_numpy(action[N * K * 2 : N * K * 2 + N * D])
        V_polar = round_to_set(V_polar, S)
        V = construct_V(M, N, D, V_polar)

        diag_polar = torch.from_numpy(action[-F:])
        diag_polar = round_to_set(diag_polar, S)
        Theta_diag = torch.polar(torch.ones(F), diag_polar)
        Theta = torch.diag(Theta_diag)

        # Calculate loss for the current action (parameters)
        is_feasible = True
        for k in range(K):
            SINR_k = SINR_k = self.objective.compute_sINR_k(
                H_T=self.objective.H_T, Theta=Theta, G=self.objective.G, V=V, W=W, k=k
            )
            if SINR_k <= self.objective.gamma[k]:
                is_feasible = True
                break

        current_loss: float = self.objective.loss(W)

        # Increment step counter
        self.steps_taken += 1

        # Calculate reward as negative loss minus constraint penalty (maximize reward = minimize loss)
        # SINR_k = self.objective.compute_sINR_k(
        #     H_T=self.objective.H_T, self.Theta=Theta, G=objective.G, V=V, W=W, k=k
        # )
        reward: float = 1 / current_loss if is_feasible else 0

        # Track best solution
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_solution = action.copy()
            self.best_params = action.copy()

        # Termination conditions
        terminated: bool = (
            current_loss < self.stop_loss
        )  # Stop if solution is sufficiently good
        truncated = self.steps_taken >= self.max_steps  # Stop if max steps reached

        # Additional info for monitoring
        info: dict = {
            "loss": current_loss,
            "best_loss": self.best_loss,
            "best_params": self.best_params,
        }

        return self.state, reward, terminated, truncated, info


register(
    id="thz",
    entry_point="env:CustomEnv",
    max_episode_steps=200,
)


def round_to_set(tensor, value_set):
    # Convert value_set to a tensor
    values = torch.tensor(list(value_set), device=tensor.device, dtype=tensor.dtype)

    # Reshape tensor for broadcasting
    tensor_expanded = tensor.unsqueeze(-1)

    # Compute absolute differences
    abs_diff = torch.abs(tensor_expanded - values)

    # Find indices of minimum differences
    min_indices = torch.argmin(abs_diff, dim=-1)

    # Get the corresponding values from the set
    result = values[min_indices]

    return result


def main():
    pass


if __name__ == "__main__":
    main()
