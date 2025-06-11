clc
clear all;
close all;

%% System model for mmWave MIMO RIS aided
global K lambda P_mean N M

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
A_1 = 6;
A_2 = 6;

M = A_1*A_2; % Number of antennas
N = 6;  % Number of RF chains
D = M/N;

% P_mean = 100:100:1000; % Power allocated for digital beamforming (dB);
P_BS_db = 12:2:30;
P_length = length(P_BS_db);
P_mean = db2pow(P_BS_db);
P_mean_init = P_mean/D;
% Number of RIS unit cells 
F_1 = 6; % = A_1
F_2 = 6; % = A_2
F = F_1*F_2;

% Number of rays per cluster
N_cl_1 = 5;
N_cl_2 = 5;
N_ray_1 = 10;
N_ray_2 = 10;

% Generate azimith and elevation angles of arrival and departure
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

% Initial digital beam matrix
W = zeros(N,K,P_length);
for p = 1:P_length
    for i = 1:N
        for j = 1:K
              W(i,j,p) =  sqrt(P_mean_init(1,p)/(K*N));
        end
    end
end

% Initial analog beam matrix
V = zeros(D*N,N,P_length);
for p = 1:P_length
for i = 1:N
    for j = ((i-1)*(D-1) + i):(i*(D-1) + i)
        if p == 1
        V(j,i,p) =  randsample(S_a,1);
        % V(j,i) = 1;
        else
        V(j,i,p) =  V(j,i,1);
        end
    end
end
end

% Create passive RIS response matrix
Omega = zeros(F,F,P_length);
beta_f = ones(F,1);
for p = 1:P_length
for i = 1:F
    if p == 1
     Omega(i,i,p) =  beta_f(i,1)*randsample(S_r,1); % Equation (4)
    else
     Omega(i,i,p) = Omega(i,i,1);
    end
end
end

%% mmWave channel model

% Generate the complex channel gain: Equation (8) and Equation (9)
G = zeros(F,M);

% Equation (8)
for i = 1:N_cl_1
    for l = 1:N_cl_1
        % Generate alpha_i_l
        alpha_i_l = sqrt(10.^(-0.1.*PL(d_BS_RIS))).* 1/2.*(randn +1i.*randn);
        % Channel from BS to RIS
        G = G + sqrt(M*F/(N_cl_1*N_ray_1)).*alpha_i_l.*a_R(phi_Rr(i,l), theta_Rr(i,l), A_1, A_2)*...
            a_B(phi_B(i,l), theta_B(i,l), A_1, A_2)';
    end
end

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
% Equation (9)

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

SINR = zeros(K,P_length);
for p = 1:P_length
for k = 1:K
    Inter = 0;
    for i = 1:K
      if i ~= k
        Inter = Inter + abs(h(:,k)'*Omega(:,:,p)*G*V(:,:,p)*W(:,i,p))^2;
      end
    end
    sigma = 10^(-3)*db2pow(sigma_dBm);
    SINR(k,p) = abs(h(:,k)'*Omega(:,:,p)*G*V(:,:,p)*W(:,k,p))^2 /(Inter + sigma); % Equation (7)
end
end

% Convert to user's data rate
R_init = zeros(K,P_length);
for p = 1:P_length
for k = 1:K
   R_init(k,p) = log(1 + SINR(k,p))/log(2);
end
end

save("RIS_Hybrid_model.mat")

%% Optimizing auxilari variables

H = zeros(K,F,P_length);
for p = 1:P_length
  H(:,:,p) = h'*Omega(:,:,p)*G;
end

H_B_R = G;

% Auxilary variable for beamforming

Q = zeros(D*N,K,P_length);
for p = 1:P_length
  Q(:,:,p) = V(:,:,p)*W(:,:,p);
end

R = zeros(K,P_length);
Inter = zeros(K,P_length);
for p = 1:P_length
for k = 1:K
    for i = 1:K
        if i ~= k
            Inter(k,p) = Inter(k,p) + abs(h(:,k)'*Omega(:,:,p)*G*V(:,:,p)*W(:,i,p))^2;
        end
    end
    R(k,p) = log(1 + abs(h(:,k)'*Omega(:,:,p)*G*V(:,:,p)*W(:,k,p))^2 /(Inter(k,p) + sigma^2))/log(2);
end
end

P_opt = zeros(1,P_length); % Power usage of beamforming scheme
% Initialize objective value
R_FD = zeros(K,P_length);
R_H = zeros(K,P_length);
R_HVb = zeros(K,P_length);
Loop = zeros(P_length,1); % Calculate number of Loop

for p = 1:P_length
obj_pre = -inf;
obj_next = geo_mean(R(:,p));

epsilon = 10^(-3); % convergence condition

gamma = zeros(K,1);
a = zeros(K,1);
b = zeros(M,K);
c = zeros(K,1);
y = zeros(K,1);

% Loop until convergence


while(abs(obj_pre - obj_next) > epsilon)
    f = 1/geo_mean(R(:,p));
    gamma = f./R(:,p);

    Psi = 0;
    for k = 1:K
        y(k,1) = Inter(k,p) + sigma^2;
        b(:,k) = H(k,:,p)'*H(k,:,p)*Q(:,k,p)/y(k,1);
        c(k,1) = abs(H(k,:,p)*Q(:,k,p))^2/(y(k,1)*(y(k,1) + abs(H(k,:,p)*Q(:,k,p))^2));
        a(k,1) = R(k,p) - abs(H(k,:,p)*Q(:,k,p))^2/y(k,1) - sigma^2*c(k,1);
        Psi = Psi + gamma(k,1)*c(k,1)*H(k,:,p)'*H(k,:,p);
    end

    P_con = 0;
    for k =1:K
        % Calculate condition for (18)
        P_con = P_con +  norm(pinv(Psi)*gamma(k,1)*b(:,k))^2;
    end

    % Update Q
    for k =1:K
       % Equation (18)
       if P_con <= P_mean(1,p)
           Q(:,k,p) = pinv(Psi)*gamma(k,1)*b(:,k);

       elseif k == 1 && P_con > P_mean(1,p)

       % Find rho by bisection
       p_a = 0;
       % Generate p_2
       p_b = p_a;
       fa = equa(p_a, Psi, gamma, b, p);
       fc = equa(p_b, Psi, gamma, b, p);
       while (fa*fc > 0)
           p_b = p_b + 0.1;
           fa = equa(p_a, Psi, gamma, b, p);
           fc = equa(p_b, Psi, gamma, b, p);
       end
    % -----------------
    % Bisection method.
    % -----------------

    p_1 = p_a;
    p_2 = p_b;
    % root estimate at first iteration
    p_mean = (p_1 + p_2)/2;

    % returns root estimate at first iteration if it is a root of f(x)
    if equa(p_mean, Psi, gamma, b, p) == 0
        rho = p_mean;
    end

    % function evaluations at first iteration
    fa = equa(p_1, Psi, gamma, b, p); 
    fc = equa(p_mean, Psi, gamma, b, p);

    % iteration
    for iter = 1:1000
        % updates interval
        if fc == 0
            break;
        elseif (fa*fc > 0)
            p_1 = p_mean;
            fa = fc;
        else
            p_2 = p_mean;
        end

        % updates root estimate
        p_mean = (p_1+p_2)/2;

        % terminates solver if converged
        if ((p_2-p_1) < 1e-10)
            break;
        end

        % function evaluation at updated root estimate
        fc = equa(p_mean, Psi, gamma, b, p);       
       end
           rho = p_mean;
           I = eye(M);
           % Update by equation (18)
           Q(:,k,p) = pinv(Psi + rho*I)*gamma(k,1)*b(:,k);
       else
           Q(:,k,p) = pinv(Psi + rho*I)*gamma(k,1)*b(:,k);
       end
    end

     % Recalculate coefficient
    b_hat_k = zeros(K,F);
    b_hat_k_1 = zeros(K,F);
    b_hat = zeros(F,1);
    c_hat = zeros(K,1);
    a_hat = zeros(K,1);
    alpha = zeros(K,K,F);
    for k = 1:K
        for n = 1:F
            Lamda_n = zeros(F,F);
            Lamda_n(n,n) = 1;
            b_hat_k(k,n) = Q(:,k,p)'*H(k,:,p)'*h(:,k)'*Lamda_n*H_B_R*Q(:,k,p);
            b_hat_k_1(k,n) = b_hat_k(k,n)/y(k,1);
            for j = 1:K
                alpha(k,j,n) = h(:,k)'*Lamda_n*H_B_R*Q(:,j,p);
            end
        end
        c_hat(k,1) = abs(H(k,:,p)*Q(:,k,p))^2/(y(k,1)*(y(k,1) + abs(H(k,:,p)*Q(:,k,p))^2));
        a_hat(k,1) = R(k,p) - abs(H(k,:,p)*Q(:,k,p))^2/y(k,1) - sigma^2*c(k,1);
    end

    for n = 1:F
        for k = 1:K
            b_hat(n,1) = b_hat(n,1) + gamma(k,1)*b_hat_k_1(k,n);
        end
    end

    Phi_k_j = zeros(F,F,K,K);
    for k = 1:K
        for j = 1:K
            for n = 1:F
                for m = 1:F
                    Phi_k_j(n,m,k,j) = alpha(k,j,n)'*alpha(k,j,m);
                end
            end
        end
    end

    Phi = 0;
    for k =1:K
        for j = 1:K
            Phi = Phi + gamma(k,1)*c_hat(k,1)*Phi_k_j(:,:,k,j);
        end
    end

    % Update phase
    for n = 1:F
        phase_sum_n = 0;
        for m = 1:F
            phase_sum_n = phase_sum_n + (1/Omega(m,m,p))*Phi(m,n);
        end
        theta_n = -angle(b_hat(n) - phase_sum_n + max(eig(Phi))*(1/Omega(n,n,p)));
        Omega(n,n,p) = exp(1i*theta_n);
    end

    % Recalculate obj value
    obj_pre = obj_next;
    R(:,p) = zeros(K,1);
    Inter(:,p) = zeros(K,1);
    for k = 1:K
      for i = 1:K
        if i ~= k
            Inter(k,p) = Inter(k,p) + abs(h(:,k)'*Omega(:,:,p)*G*Q(:,i,p))^2;
        end
      end
    R(k,p) = log(1 + abs(h(:,k)'*Omega(:,:,p)*G*Q(:,k,p))^2 /(Inter(k,p) + sigma^2))/log(2);
    end

    obj_next = geo_mean(R(:,p));
    Loop(p) = Loop(p) + 1;
    P_opt(:,p) = norm(Q(:,:,p),'fro')^2;
end

% Remapping Omega into S_r
for i = 1:F
   S_r_round = S_r(1,1);
   min_dist = abs(Omega(i,i,p) - S_r_round);
   for q = 1:2^Q_2
      if abs(Omega(i,i,p) - S_r(q,1)) < min_dist
         S_r_round = S_r(q,1);
         min_dist = abs(Omega(i,i,p) - S_r(q,1));
      end
   end
   Omega(i,i,p) = S_r_round;
end
end

for p = 1:P_length
  for k = 1:K
      for i = 1:K
        if i ~= k
            Inter(k,p) = Inter(k,p) + abs(h(:,k)'*Omega(:,:,p)*G*Q(:,i,p))^2;
        end
      end
    R_FD(k,p) = log(1 + abs(h(:,k)'*Omega(:,:,p)*G*Q(:,k,p))^2 /(Inter(k,p) + sigma^2))/log(2);
   end
end
save("RIS_Hybrid_PreOpt.mat")

% Extract Q = V*W
for p = 1:P_length
Q_hat = reshape(Q(:,:,p), [D,N,K]);
for k = 1:K
    for i = 1:N
        q_i = Q_hat(:,i,k);
        % Update v_i
        V(((i-1)*(D-1) + i):(i*(D-1) + i),i,p) = exp(1i*phase(q_i));
        % Update w_i
        W(i,k,p) = sum(abs(q_i))/D;
    end
end

R_H(:,p) = zeros(K,1);
Inter(:,p) = zeros(K,1);
 for k = 1:K
    for i = 1:K
       if i ~= k
           Inter(k,p) = Inter(k,p) + abs(h(:,k)'*Omega(:,:,p)*G*V(:,:,p)*W(:,i,p))^2;
       end
    end
    R_H(k,p) = log(1 + abs(h(:,k)'*Omega(:,:,p)*G*V(:,:,p)*W(:,k,p))^2 /(Inter(k,p) + sigma^2))/log(2);
 end

% Remapping V into S_a
for i = 1:N
    for j = ((i-1)*(D-1) + i):(i*(D-1) + i)
        S_a_round = S_a(1,1);
        min_dist = abs(V(j,i,p) - S_a_round);
        for q = 1:2^Q_1
            if abs(V(j,i,p) - S_a(q,1)) < min_dist
                S_a_round = S_a(q,1);
                min_dist = abs(V(j,i,p) - S_a(q,1));
            end
        end
        V(j,i,p) = S_a_round;
    end
end

P_opt(:,p) = norm(V(:,:,p)*W(:,:,p),'fro')^2;
% Recalculate obj value
R_HVb(:,p) = zeros(K,1);
Inter(:,p) = zeros(K,1);
 for k = 1:K
    for i = 1:K
       if i ~= k
           Inter(k,p) = Inter(k,p) + abs(h(:,k)'*Omega(:,:,p)*G*V(:,:,p)*W(:,i,p))^2;
       end
    end
    R_HVb(k,p) = log(1 + abs(h(:,k)'*Omega(:,:,p)*G*V(:,:,p)*W(:,k,p))^2 /(Inter(k,p) + sigma^2))/log(2);
 end
end

save("RIS_Hybrid_Opt.mat")