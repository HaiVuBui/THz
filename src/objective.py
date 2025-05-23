import torch
import numpy as np
from dataclasses import dataclass
import math


def construct_V(M: int, N: int, D: int, V_polar: torch.Tensor) -> torch.Tensor:
    """Construct a tensor V based on phase angles V_polar."""
    V = torch.zeros((M, N), dtype=torch.cfloat)
    for n in range(N):
        magnitudes = torch.ones(D)
        v_n = torch.polar(magnitudes, V_polar[n])
        V[n * D : n * D + D, n] = v_n
    return V


@dataclass
class Objective:
    """Class to compute objective and SINR for a given system configuration."""

    M: int
    N: int
    D: int
    K: int
    F: int
    G: torch.Tensor
    H_T: torch.Tensor
    sigma2: torch.Tensor
    gamma: torch.Tensor
    S: list[float]

    def loss(self, W: torch.Tensor) -> float:
        """Compute the objective value based on W and D."""
        D = self.D
        if isinstance(W, np.ndarray):
            W = torch.from_numpy(W)
        obj = D * torch.sum(torch.abs(W) ** 2)
        return obj.item()

    def compute_sINR_k(
        self,
        H_T: torch.Tensor,
        Theta: torch.Tensor,
        G: torch.Tensor,
        V: torch.Tensor,
        W: torch.Tensor,
        k: int,
    ) -> float:
        """Compute the SINR for user k."""
        # args = [H_T, Theta, G, V, W]
        # for arg in args:
        #     if isinstance(arg, np.ndarray):
        #         arg = torch.from_numpy(arg)

        numerator = torch.abs(H_T[k] @ Theta @ G @ V @ W.T[k])

        H_T_no_k = torch.cat((H_T[:k], H_T[k + 1 :]), dim=0)

        interference = torch.sum(torch.abs(H_T_no_k @ Theta @ G @ V @ W.T[k]) ** 2)
        denominator = interference + self.sigma2[k]

        return numerator.item() / denominator.item()

    def action_shape(self):
        return self.N * self.K * 2 + self.N + self.F

    def state_shape(self):
        return 0


def make_obj(M: int = 4, N: int = 2, K: int = 2, F: int = 3):
    # System dimensions
    D = M // N

    # Channel matrices
    G = torch.randn(F, M, dtype=torch.cfloat)
    H_T = torch.randn(K, F, dtype=torch.cfloat)

    # Noise and threshold vectors
    sigma2 = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32)
    gamma = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    S_num = 4
    S = [math.pi * 2 * i / S_num for i in range(S_num)]

    # Create objective instance
    objective = Objective(
        M=M, N=N, D=D, K=K, F=F, G=G, H_T=H_T, sigma2=sigma2, gamma=gamma, S=S
    )
    return objective


def main():
    M = 8
    N = 4
    K = 2
    F = 3
    D = M // N
    objective = make_obj(M=M, N=N, K=K, F=F)

    action = torch.rand(N * K * 2 + N * D + F)

    # Precoding matrix W
    W_module = action[: N * K]
    W_polar = action[N * K : 2 * N * K]
    # return

    W = torch.polar(W_module, W_polar).reshape(N, K)

    # Beamforming matrix V
    V_polar = action[N * K * 2 : N * K * 2 + N * D]
    V = construct_V(M, N, D, V_polar)

    # Diagonal phase matrix Theta
    diag_polar = action[-F:]
    Theta_diag = torch.polar(torch.ones(F), diag_polar)
    Theta = torch.diag(Theta_diag)

    # Compute and print results
    obj_value = objective.loss(W)

    SINR_k = objective.compute_sINR_k(
        H_T=objective.H_T, Theta=Theta, G=objective.G, V=V, W=W, k=0
    )

    return
    print(f"Objective value: {obj_value}")
    for k in range(objective.K):
        SINR_k = objective.compute_sINR_k(
            H_T=objective.H_T, Theta=Theta, G=objective.G, V=V, W=W, k=k
        )
        print(f"SINR for user {k}: {SINR_k}")

    print(action)
    print(W_module)
    print(W_polar)
    print(V_polar)
    print(diag_polar)


if __name__ == "__main__":
    main()
