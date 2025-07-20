clc
clear all;
close all;

%% System model for mmWave MIMO RIS aided
global K lambda

K = 3; % Number of users
% Coordinates of users, BS and RIS (Fixed BS, users)
BS_coor = [0,0]; % BS coordinate (0m,0m)
user_coor = zeros(K,2); % Init users' coordinate

x0 = 100; % Center of the circle in the x direction.
y0 = 0; % Center of the circle in the y direction.

% Now create the set of points.
t = 2*pi*rand(K,1);
r = 5; % Circle diameter

x = x0 + r.*cos(t); % random x coordinate
y = y0 + r.*sin(t); % random y coordinate
% Users coordinate
user_coor(:,1) = x;
user_coor(:,2) = y;

% RIS coordinate
d_RIS = 50;
RIS_coor = [d_RIS,10]; % (d_RISm, 10)

% Calculate distances
d_BS_RIS = norm(BS_coor - RIS_coor); % Distance from BS to RIS
d_RIS_UE = zeros(K,1);
for k =1:K
    d_RIS_UE(k,1) = norm(user_coor(k,:) - RIS_coor);
end

% Operation parameters
f_c = 28*10^9; % Operate frequency 
lambda = (3*10^8)/f_c; % Operate wavelength
W = 251.1886*10^6; % Operate Bandwidth
sigma_dBm = -174 + pow2db(W); % Noise variance with power of -90 dBm

% UPA array with a 6x6 structure
A_1 = 8;
A_2 = 8;

M = A_1*A_2; % Number of antennas
N = 8;  % Number of RF chains
D = M/N;

sigma_P = 10; % Power deviation
P_mean = 1000; % Power allocated for digital beamforming


% Number of RIS unit cells 
F_1 = 8; % = A_1
F_2 = 8; % = A_2
F = F_1*F_2;

% Number of rays per cluster
N_cl_1 = 5;
N_cl_2 = 5;
N_ray_1 = 10;
N_ray_2 = 10;

% Generate azumith and elevation angles of arrival and departure
% with angle spread of 10 degree
sigma_angle = sqrt(deg2rad(10));

phi_Rr = sigma_angle*randl(N_cl_1, N_ray_1);
theta_Rr = sigma_angle*randl(N_cl_1, N_ray_1);
phi_B = sigma_angle*randl(N_cl_1, N_ray_1);
theta_B = sigma_angle*randl(N_cl_1, N_ray_1);

phi_Rt = sigma_angle*randl(N_cl_2, N_ray_2);
theta_Rt = sigma_angle*randl(N_cl_2, N_ray_2);

%% RIS and Beamforming model
Q_1 = 3; % Number of analog phase shifter control bit
Q_2 = 3; % Number of each RIS element control bit

S_a = zeros(2^Q_1,1); % set of all possible phase shifts for analog beamformer
S_r = zeros(2^Q_2,1); % set of all possible RIS reflection coefficient

for i = 1:2^Q_1
    S_a(i,1) = exp(1i*2*pi*(i-1)/2^Q_1); % Equation (1)
end

for i = 1:2^Q_2
    S_r(i,1) = exp(1i*2*pi*(i-1)/2^Q_2); % Equation (5)
end

% Create digital beam matrix
W = zeros(N,K);
for i = 1:N
    for j = 1:K
        W(i,j) =  P_mean + sqrt(sigma_P/2)*(rand + 1i*rand);
    end
end

% Create analog beam matrix
V = zeros(D*N,N);
for i = 1:N
    for j = ((i-1)*(D-1) + i):(i*(D-1) + i)
        V(j,i) =  randsample(S_a,1);
        % V(j,i) = 1;
    end
end

% Create passive RIS response matrix
Omega = zeros(F,F);
beta_f = ones(F,1);
for i = 1:F
     Omega(i,i) =  beta_f(i,1)*randsample(S_r,1); % Equation (4)
end

%% mmWave channel model

% Generate the complex channel gain 
G = zeros(F,M);
% Equation (8)
for i = 1:N_cl_1
    for l = 1:N_ray_1
        % Generate alpha_i_l
        alpha_i_l = sqrt(10.^(-0.1.*PL(d_BS_RIS))).* 1/2.*(randn +1i.*randn);
        % Channel from BS to RIS
        a_test = sqrt(M*F/(N_cl_1*N_ray_1)).*alpha_i_l.*a_R(phi_Rr(i,l), theta_Rr(i,l), A_1, A_2)*...
            a_B(phi_B(i,l), theta_B(i,l), A_1, A_2)';
        G = G + sqrt(M*F/(N_cl_1*N_ray_1)). * alpha_i_l.*a_R(phi_Rr(i,l), theta_Rr(i,l), A_1, A_2)*...
            a_B(phi_B(i,l), theta_B(i,l), A_1, A_2)';
    end
end

% Equation (9)
h = zeros(F,K);
for k = 1:K
    for i = 1:N_cl_2
        for l = 1:N_ray_2
            %Generate beta_i_l
            beta_i_l = sqrt(10^(-0.1.*PL(d_RIS_UE(k,1)))).* 1/2.*(randn +1i.*randn);
            % Channel from RIS to user k
            h(:,k) = h(:,k) + sqrt(F/(N_cl_2*N_ray_2))*beta_i_l*a_R(phi_Rt(i,l), theta_Rt(i,l), A_1, A_2);
        end
    end
end

%% Calculate SINR of users

SINR = zeros(K,1);
for k = 1:K
    Inter = 0;
    for i = 1:K
      if i ~= k
        Inter = Inter + abs(h(:,k)'*Omega*G*V*W(:,i))^2;
      end
    end
    sigma = 10^(-3)*db2pow(sigma_dBm);
    SINR(k,1) = abs(h(:,k)'*Omega*G*V*W(:,k))^2 /(Inter + sigma); % Equation (7)
end

save("RIS_Hybrid_model.mat")
