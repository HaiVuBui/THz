import torch
import numpy as np

class LeastSquares:
    """
    Represents a least squares optimization problem of the form:
    min ||Ax - b||^2

    This class provides methods to calculate the loss, solution, and constraints
    for the least squares problem.
    """

    def __init__(self, shape=(100, 100), device=None):
        """
        Initialize a least squares problem with random A and b.

        Args:
            shape (tuple): Shape of the A matrix (m, n)
            device (str): Device to use for torch tensors ('cpu' or 'cuda')
        """
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.A = torch.rand(*shape, dtype=torch.float32, device=self.device)
        self.b = torch.rand(shape[0], dtype=torch.float32, device=self.device)

    def solution(self):
        """
        Calculate the analytical solution x* = (A^T A)^(-1) A^T b using stable methods.

        Returns:
            torch.Tensor: The solution vector x*
        """
        try:
            return torch.linalg.solve(self.A, self.b)
        except RuntimeError:
            # Fallback to pseudoinverse for ill-conditioned matrices
            return torch.linalg.pinv(self.A) @ self.b

    def action_shape(self):
        """
        Get the shape of the action space (dimension of x).

        Returns:
            int: Dimension of the solution vector x
        """
        return self.A.shape[1]

    def state_shape(self):
        """
        Get the shape of the state space (flattened A and b).

        Returns:
            int: Total dimension of the state vector
        """
        return self.A.shape[0] * self.A.shape[1] + self.A.shape[0]

    def loss(self, x):
        """
        Calculate the loss ||Ax - b||^2 for a given x.

        Args:
            x (numpy.ndarray or torch.Tensor): Solution vector x

        Returns:
            float: The loss value
        """
        # Convert x to tensor if it's a numpy array
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        elif x.device != self.device:
            x = x.to(self.device)

        # Use torch functions for computation
        return torch.norm(self.A @ x - self.b).item()

    def constraints(self, x):
        """
        Calculate constraint violations for the solution x.
        This is a placeholder for potential constraints on the solution.

        Args:
            x (numpy.ndarray or torch.Tensor): Solution vector x

        Returns:
            float: Penalty value for constraint violations (0 if no violations)
        """
        # Example constraint: L1 regularization
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        elif x.device != self.device:
            x = x.to(self.device)

        # No constraints by default
        return 0.0

    def to(self, device):
        """
        Move the problem to a specific device.

        Args:
            device (str): Device to move tensors to ('cpu' or 'cuda')

        Returns:
            LeastSquares: Self for chaining
        """
        self.device = device
        self.A = self.A.to(device)
        self.b = self.b.to(device)
        return self