from numpy.random import beta
import torch
import numpy as np
import numpy.linalg as nla
from dataclasses import dataclass
import math

from torch._C import dtype


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
        return self.F * self.M * 2 + self.K * self.F * 2 + self.N * self.K * 2 + self.N + self.F
 
    
    def reset_state(self):
        F = self.F
        M = self.M
        K = self.K
        self.G = torch.randn(F, M, dtype=torch.cfloat)
        self.H_T = torch.randn(K, F, dtype=torch.cfloat)
        return torch.concatenate([torch.abs(self.G).reshape(-1),
                                  torch.angle(self.G).reshape(-1),
                                  torch.abs(self.H_T).reshape(-1),
                                  torch.angle(self.H_T).reshape(-1)]) 



def pow2db(W):
    return 10 * np.log10(W)

def make_obj(M: int = 4, N: int = 2, K: int = 2, F: int = 3):
    

    # Channel matrices
    G = torch.randn(F, M, dtype=torch.cfloat)
    H_T = torch.randn(K, F, dtype=torch.cfloat)

    # Noise and threshold vectors
    sigma2 = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32)
    gamma = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    S_num = 4
    S = [math.pi * 2 * i / S_num for i in range(S_num)]

    # Create ojective instance
    objective = Objective(
        M=M, N=N, D=D, K=K, F=F, G=G, H_T=H_T, sigma2=sigma2, gamma=gamma, S=S
    )
    return objective

def test():
    K = 3
    # System dimensions
    BS_coor = np.zeros(2, dtype = float)
    user_coor = np.zeros((K,2), dtype = float)

    # center
    x0 = 100.
    y0 = 0.
    
    # radius and phase
    t = 2 * np.pi * np.random.rand(K)
    r = 5

    # random x and y coor
    x = x0 * r * np.cos(t)
    y = y0 + r * np.sin(t)

    user_coor.T[0] = x
    user_coor.T[1] = y

    # Ris coor
    d_Ris = 50
    RIS_coor = np.array([d_Ris, 10])

    # Distance
    d_BS_Ris = nla.norm(BS_coor - RIS_coor)
    d_RIS_user = np.zeros(K)
    for i, d in enumerate(d_RIS_user):
        d_RIS_user[i] = nla.norm(user_coor[i] - RIS_coor)
    print(d_RIS_user)

    # operation parameters
    f_c = 28 * 10**9
    Lambda = (3 * 10**8)/f_c
    W = 251.1884 * 10**6
    sigma_dBm = -174 + pow2db(W)

    #
    A_1 = 8
    A_2 = 8
    M = A_1 * A_2
    N = 8
    D = M // N

    sigma_P = 10
    P_mean = 1000

    F_1 = 8
    F_2 = 8

    F = F_1 * F_2

    N_cl_1 = 5
    N_cl_2 = 5
    N_ray_1 = 10
    N_ray_2 = 10
    
    sigma_angle = np.sqrt(10/180 * np.pi)

    phi_Rr = sigma_angle * np.random.rand(N_cl_1, N_ray_1)
    theta_Rr  = sigma_angle * np.random.rand(N_cl_1, N_ray_1)
    phi_B = sigma_angle * np.random.rand(N_cl_1, N_ray_1)
    theta_B  = sigma_angle * np.random.rand(N_cl_1, N_ray_1)

    phi_Rt = sigma_angle * np.random.rand(N_cl_2, N_ray_2)
    theta_Rt = sigma_angle * np.random.rand(N_cl_2, N_ray_2)

    # RIS and beamforming model
    Q_1 = 3
    Q_2 = 3

    S_a_phases = torch.tensor([i/(2**Q_1) for i in range(2**Q_1)])
    S_r_phases = torch.tensor([i/(2**Q_2) for i in range(2**Q_2)])

    S_a = torch.polar(torch.ones(S_a_phases.shape), S_a_phases)
    S_r = torch.polar(torch.ones(S_r_phases.shape), S_r_phases)

    W_real = torch.rand(N,K)
    W_im = torch.rand(N,K)
    W = P_mean + math.sqrt(sigma_P/2)  * torch.complex(W_real, W_im)

    V_polar = S_r_phases[torch.randint(len(S_r_phases), (N,D))] 
    V = construct_V(M, N, D, V_polar)

    beta_f = torch.ones(F)
    Omega_polar = S_r_phases[torch.randint(len(S_r_phases), (F,))] 
    Omega_diag = torch.polar(beta_f, Omega_polar)
    Omega = torch.diag(Omega_diag)
 


    # G
    def a_z(phi, delta):
        phase = torch.zeros(A_1 * A_2)
        idx = 0
        for p in range(A_2):
            for o in range(A_1):
                phase[idx] = 2 * np.pi * (o * np.sin(phi) * np.sin(delta) + p * np.cos(delta))
                idx += 1
        out = (1/np.sqrt(A_1 * A_2)) * torch.exp(1j * phase)
        return out

    def PL(d):
        out = 72. + 10 * 2.92 * np.log10(d)
        return out

    G = torch.zeros((F,M), dtype = torch.cfloat)
    alpha = np.sqrt(10**(-0.1 * PL(d_BS_Ris))) * 0.5 * torch.rand((N_cl_1,N_ray_1), dtype=torch.cfloat)
    for i in range(N_cl_1):
        for l in range(N_ray_1):
            a_R = a_z(phi_Rr[i,l], theta_Rr[i,l])
            a_B = a_z(phi_B[i,l], theta_B[i,l])

            G += np.sqrt((M*F)/(N_cl_1 * N_ray_1)) * alpha[i,l] * torch.outer(a_R,torch.conj(a_B))

    # H_T
    H_T = torch.zeros((K,F), dtype = torch.cfloat)
    for k,H_T_k in enumerate(H_T):
        for i in range(N_cl_2):
            for l in range(N_ray_2):
                beta_k = np.sqrt(10**(-0.1 * PL(d_RIS_user[k]))) * 0.5 * torch.rand((N_cl_2, N_ray_2), dtype=torch.cfloat)
                H_T[k] += np.sqrt(F/(N_cl_2 * N_ray_2)) * beta_k[i,l] * a_z(phi_Rt[i,l], theta_Rt[i,l])

    # SINR
    db2pow = lambda ydb: 10**(ydb/10)

    SINR = np.zeros(K)
    sigma = db2pow(sigma_dBm)
    for k in range(K):
        numer = abs(H_T[k] @ Omega @ G @ V @ W.T[k]) ** 2
        H_T_no_k = torch.cat((H_T[:k],H_T[k+1:]))
        denor = torch.norm(H_T_no_k @ Omega @ G @ V @ W.T[k])**2 + sigma
        SINR[k] = numer/denor
    print(SINR)




                





    print('no bug')

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
    
    T = torch.rand(objective.state_shape())
    ac = objective.reset_state()
    print(ac.shape)
    print(T[: F * M *2 + F * K * 2 ].shape)
    T[: F * M *2 + F * K * 2 ] = ac 

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
    # main()
    test()
