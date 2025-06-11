function sum = equa(rho, Psi, gamma, b)
global K N P
I = eye(N);
sum = 0;
for k =1:K
  sum = sum + norm(pinv(Psi + rho*I)*gamma(k,1)*b(:,k))^2;
end
sum = sum - P;
end

