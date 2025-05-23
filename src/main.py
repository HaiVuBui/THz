from operator import le
import torch
from ddpg import train_ddpg
from env import CustomEnv
from objective import Objective, construct_V
from LQ import LeastSquares


def main():
    """Main function to demonstrate objective and SINR computation."""
    # System dimensions
    M, N, K, F = 4, 2, 2, 3
    D = M // N

    # Channel matrices
    G = torch.randn(F, M, dtype=torch.cfloat)
    HT = torch.randn(K, F, dtype=torch.cfloat)

    # Noise and threshold vectors
    sigma2 = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32)
    gamma = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

    # Beamforming matrix V
    V_thetas = torch.rand(N, D)
    V = construct_V(M, N, D, V_thetas)

    # Precoding matrix W
    W = torch.randn(N, K, dtype=torch.cfloat)

    # Diagonal phase matrix Theta
    diag_thetas = torch.rand(F)
    Theta_diag = torch.polar(torch.ones(F), diag_thetas)
    Theta = torch.diag(Theta_diag)

    # Create objective instance
    objective = Objective(
        M=M, N=N, D=D, K=K, F=F, G=G, HT=HT, sigma2=sigma2, gamma=gamma
    )

    # Compute and print results
    obj_value = objective.compute_objective(W, D)

    print(f"Objective value: {obj_value}")
    for k in range(K):
        sinr_k = objective.compute_sINR_k(HT=HT, Theta=Theta, G=G, V=V, W=W, k=k)
        print(f"SINR for user {k}: {sinr_k}")

    # env = CustomEnv(objective=objective)
    obj = LeastSquares((5, 5))
    env = CustomEnv(obj)

    agent, _ = train_ddpg(
        env, num_episodes=1000, max_steps=1000, batch_size=256, hidden_dim=512
    )


if __name__ == "__main__":
    main()
