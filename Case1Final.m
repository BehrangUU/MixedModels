%#------------------------------------------------------------------#
%# Function   :  Fitting linear mixed model - Case 1                #
%# Written by :  Behrang Mahjani                                    #
%# Created on :  12/16/2015                                         #
%#------------------------------------------------------------------#
%% Initialization
clear all
%load('Ainv.mat'); % Load the data Ainv

%Or creat some positive definie symmetic covariance matrix
q = 1000;
A = sprandsym(q,0.01);
A = A'+A + q*eye(q);

[q,~] = size(A);
%q = 800; %Cut part of it
Ainv = A(1:q,1:q);

n = q;
p = 1;
k = 70;

N = 10; % number of iterations

%--- simulate y
%A = inv(Ainv); % For simulating y
R = chol(Ainv);
I2 = speye(size(Ainv));
A = R\(R'\I2); %inverse of A

Ainv = sparse(Ainv);
AinvSqrt = chol(Ainv);
ASqrt = inv(AinvSqrt);

sigma_e = sqrt(2);
sigma_u = sqrt(1);

Z = eye(q);
X = [zeros(n,1),zeros(n,p-1)];

MU = zeros(q,1)';
SIGMA = sigma_u^2*A;
V = mvnrnd(MU,SIGMA);
u = V';
a = AinvSqrt'*u;

e = mvnrnd(zeros(n,1)',eye(n)*sigma_e^2)';
y =  a + e;
%% Initilization - part 2
sigma_e = sqrt(1);
sigma_u = sqrt(0.9);

sigma2_e_theory = sigma_e^2;
sigma2_e_lanc = sigma_e^2;
sigma2_u_theory = sigma_u^2;
sigma2_u_lanc = sigma_u^2;

se2_theory = zeros(1,N);
se2_lanc = zeros(1,N);
su2_lanc = zeros(1,N);
su2_theory = zeros(1,N);

u_theory = zeros(q,N);
u_lanc = zeros(q,N);

se2_theory(1,1) = sigma2_e_theory;
se2_lanc(1,1) = sigma2_e_lanc;
su2_theory(1,1) = sigma2_u_theory;
su2_lanc(1,1) = sigma2_u_lanc;

%% DIRECT method
tic
for i = 1:N
    LHS = eye(q) + se2_theory(1,i)/su2_theory(1,i)*Ainv;
    %invLHS = inv(LHS);
    
    R = chol(LHS);
    I2 = speye(size(LHS));
    invLHS = R\(R'\I2);
    
    u_theory(:,i) = invLHS*y;
    
    u_y = invLHS;
    H_u = eye(q)-AinvSqrt*invLHS*ASqrt';
    H_beta = invLHS;
    
    e = y - u_theory(:,i);
    
    v = Ainv*u_theory(:,i);
    se2_theory(1,i+1) = (sum(e.^2)+se2_theory(1,i)*trace(H_beta))/n;
    su2_theory(1,i+1) = (sum(v) + su2_theory(1,i)*trace(H_u))/q;
    
    disp([i+1,se2_theory(1,i+1),su2_theory(1,i+1)])
end
t = toc;

%% Lanczos
tic
r0 = y;
[Uk,T] = LAN(Ainv,k,r0);
for i = 1:N
    %invRHS = inv(eye(k) + se2_lanc(1,i)/su2_lanc(1,i)*T);
    LHS = eye(k) + se2_lanc(1,i)/su2_lanc(1,i)*T;
    [L,U,P] = lu(LHS);
    I = eye(size(LHS));
    invLHS = zeros(size(LHS));
    invLHS(:,1:k) = U\(L\I(:,1:k));
    
    %      R=chol(LHS);
    %      I2=speye(size(LHS));
    %      invLHS=R\(R'\I2);
    
    P1 = Uk*invLHS;
    P2 = Uk'*y;
    u_lanc(:,i) = P1*P2;
    
    T_Hbeta = sparsetrace(Uk,invLHS*Uk');% trace of H Beta
    T_Hu = q - T_Hbeta;
    
    e = y - u_lanc(:,i);
    v = Ainv*u_lanc(:,i);
    se2_lanc(1,i+1) = (sum(e.^2)+se2_lanc(1,i)*T_Hbeta)/n;
    su2_lanc(1,i+1) = (sum(v) + su2_lanc(1,i)*T_Hu)/q;
    
    disp([i+1,se2_theory(1,i+1),su2_theory(1,i+1),se2_lanc(1,i+1),su2_lanc(1,i+1)])
end
t2 = toc;

disp([t,t2])