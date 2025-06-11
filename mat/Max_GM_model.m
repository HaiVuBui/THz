clc
clear all;
close all;

% Fixing all parameters (Preventing numerical result)

%% System model for mmWave MIMO RIS aided
global K N P

K = 3; % Number of users

% UPA array with a 6x6 structure
A_1 = 6;
A_2 = 6;

M = A_1*A_2; % Number of antennas
N = 6;  % Number of RF chains
D = M/N;

P = 10^6; % Power constraint

% Number of RIS unit cells 
F_1 = 6; % = A_1
F_2 = 6; % = A_2
F = F_1*F_2;

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

% Fix analog beam matrix
V = zeros(D*N,N);
for i = 1:N
    for j = ((i-1)*(D-1) + i):(i*(D-1) + i)
        V(j,i) =  randsample(S_a,1);
        % V(j,i) = 1;
    end
end

%% mmWave channel model

% Generate the complex channel gain 
G = zeros(F,M);
% Equation (8)
for i = 1:F
    for j = 1:M
        % G(i,j) =  1/2.*(randn +1i.*randn);
        G(i,j) =  0.1;
    end
end

% Equation (9)
h = zeros(F,K);
for k = 1:K
    for j = 1:F
            % h(j,k) =  1/2.*(randn +1i.*randn);
            h(j,k) =  1;
    end
end

%% Solving GM optimization problem

% Initilize digital beamforming and RIS phase shift
W = zeros(N,K);
for i = 1:N
    for j = 1:K
        W(i,j) = (rand + 1i*rand);
    end
end

Omega = zeros(F,F);
for i = 1:F
     Omega(i,i) = exp(1i*2*pi*rand);
end

H = h'*Omega*G*V;

H_B_R = G*V;

% Objective function value
sigma = rand;

R = zeros(K,1);
Inter = zeros(K,1);
for k = 1:K
    for i = 1:K
        if i ~= k
            Inter(k,1) = Inter(k,1) + abs(h(:,k)'*Omega*G*V*W(:,i))^2;
        end
    end
    R(k,1) = log(1 + abs(h(:,k)'*Omega*G*V*W(:,k))^2 /(Inter(k,1) + sigma^2));
end

% Initialize objective value
obj_pre = -inf;
obj_next = geo_mean(R);

epsilon = 10^(-5); % convergence condition

gamma = zeros(K,1);
a = zeros(K,1);
b = zeros(N,K);
c = zeros(K,1);
y = zeros(K,1);

% Loop until convergence
Loop = 0; % Calculate number of Loop

while(abs(obj_pre - obj_next) > epsilon)
    f = 1/geo_mean(R);
    gamma = f./R;

    Psi = 0;
    for k = 1:K
        y(k,1) = Inter(k,1) + sigma^2;
        b(:,k) = H(k,:)'*H(k,:)*W(:,k)/y(k,1);
        c(k,1) = abs(H(k,:)*W(:,k))^2/(y(k,1)*(y(k,1) + abs(H(k,:)*W(:,k))^2));
        a(k,1) = R(k,1) - abs(H(k,:)*W(:,k))^2/y(k,1) - sigma^2*c(k,1);
        Psi = Psi + gamma(k,1)*c(k,1)*H(k,:)'*H(k,:);
    end

    P_con = 0;
    for k =1:K
        % Calculate condition for (18)
        P_con = P_con +  norm(pinv(Psi)*gamma(k,1)*b(:,k))^2;
    end

    % Update W
    for k =1:K
       % Equation (18)
       if P_con <= P
           W(:,k) = pinv(Psi)*gamma(k,1)*b(:,k);
       else

       % Find rho by bisection
       p_a = 0;
       % Generate p_2
       p_b = p_a;
       fa = equa(p_a, Psi, gamma, b);
       fc = equa(p_b, Psi, gamma, b);
       while (fa*fc > 0)
           p_b = p_b + 0.1;
           fa = equa(p_a, Psi, gamma, b);
           fc = equa(p_b, Psi, gamma, b);
       end
    % -----------------
    % Bisection method.
    % -----------------

    p_1 = p_a;
    p_2 = p_b;
    % root estimate at first iteration
    p_mean = (p_1 + p_2)/2;

    % returns root estimate at first iteration if it is a root of f(x)
    if equa(p_mean, Psi, gamma, b) == 0
        rho = p_mean;
    end

    % function evaluations at first iteration
    fa = equa(p_1, Psi, gamma, b); 
    fc = equa(p_mean, Psi, gamma, b);

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
        fc = equa(p_mean, Psi, gamma, b);       
    end
           rho = p_mean;
           I = eye(N);
           % Update by equation (18)
           W(:,k) = pinv(Psi + rho*I)*gamma(k,1)*b(:,k);
       end
    end

    % Recalculate coefficient
    b_hat = zeros(K,F);
    c_hat = zeros(K,1);
    a_hat = zeros(K,1);
    alpha = zeros(K,K,F);
    for k = 1:K
        for n = 1:F
            Lamda_n = zeros(F,F);
            Lamda_n(n,n) = 1;
            b_hat(k,n) = W(:,k)'*H(k,:)'*h(:,k)'*Lamda_n*H_B_R*W(:,k);
            for j = 1:K
                alpha(k,j,n) = h(:,k)'*Lamda_n*H_B_R*W(:,j);
            end
        end
        c_hat(k,1) = abs(H(k,:)*W(:,k))^2/(y(k,1)*(y(k,1) + abs(H(k,:)*W(:,k))^2));
        a_hat(k,1) = R(k,1) - abs(H(k,:)*W(:,k))^2/y(k,1) - sigma^2*c(k,1);
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
            phase_sum_n = phase_sum_n + (1/Omega(m,m))*Phi(m,n);
        end
        theta_n = -angle(b_hat(n) - phase_sum_n + max(eig(Phi))*(1/Omega(n,n)));
        Omega(n,n) = exp(1i*theta_n);
    end

    % Recalculate obj value
    obj_pre = obj_next;
    for k = 1:K
    for i = 1:K
        if i ~= k
            Inter(k,1) = Inter(k,1) + abs(h(:,k)'*Omega*G*V*W(:,i))^2;
        end
    end
    R(k,1) = log(1 + abs(h(:,k)'*Omega*G*V*W(:,k))^2 /(Inter(k,1) + sigma^2));
    end
    obj_next = geo_mean(R);
    Loop = Loop + 1;
    P_opt = norm(W,'fro')^2;
end
