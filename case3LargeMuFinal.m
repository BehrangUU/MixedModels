%#------------------------------------------------------------------#
%# Function   :  Fitting linear mixed model - Case 3, Large mu      #
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
R = chol(Ainv);
I2 = speye(size(Ainv));
A = R\(R'\I2); % inverse of A
     
Ainv = sparse(Ainv);  % save it as a sparse matrix

AinvSqrt = chol(Ainv);

R = chol(AinvSqrt);
I2 = speye(size(AinvSqrt));
ASqrt  =R\(R'\I2);

sigma_e = sqrt(2); 
sigma_u = sqrt(1);

Z = kron(eye(q),ones(n/q,1));
X = [ones(n,1),rand(n,p-1)];

MU = zeros(q,1)';
SIGMA = sigma_u^2*A;
V = mvnrnd(MU,SIGMA);
u = V';
a = AinvSqrt'*u;
beta = rand(p,1);

e = mvnrnd(zeros(n,1)',eye(n)*sigma_e^2)';

y = X*beta + Z*a + e;
real_beta = beta;
real_u = u;

Xa = [X,Z;zeros(q,p),AinvSqrt];
ya = [y;zeros(q,1)];
%% Initilization - Part 2
sigma_e = sqrt(2); 
sigma_u = sqrt(0.2);

sigma2_e_theory = sigma_e^2;   
sigma2_e_lanc = sigma_e^2;
sigma2_u_theory = sigma_u^2;
sigma2_u_lanc = sigma_u^2;

se2_theory = zeros(1,N);
se2_lanc = zeros(1,N);
su2_lanc = zeros(1,N);
su2_theory = zeros(1,N);
se2_lanc_small = zeros(1,N);
su2_lanc_small = zeros(1,N);

u_theory = zeros(q,N);
u_lanc = zeros(q,N);
u_lanc_small = zeros(q,N);

se2_theory(1,1) = sigma2_e_theory;
se2_lanc(1,1) = sigma2_e_lanc;    
se2_lanc_small(1,1) = sigma2_e_lanc; 
su2_theory(1,1) = sigma2_u_theory;
su2_lanc(1,1) = sigma2_u_lanc;
su2_lanc_small(1,1) = sigma2_u_lanc;

%% DIRECT
 tic
for i = 1:N
    
    W = [(1/se2_theory(1,i))*eye(n),zeros(n,q);zeros(q,n),(1/su2_theory(1,i))*eye(q)];
    
    LHS = (Xa'*W*Xa);
    R = chol(LHS);
    I = eye(size(LHS));
    invLHS = R\(R'\I);
     
%    beta_a = (Xa'*W*Xa)\(Xa'*W*ya);
    beta_a = invLHS*(Xa'*W*ya); %inv(Xa'*W*Xa)*Xa'*W*ya;

    beta_theory = beta_a(1:p);
    u_theory(:,i) = beta_a((p+1):end);

%    H = Xa*((Xa'*W*Xa)\(Xa'*W));
    H = Xa*((invLHS)*(Xa'*W));
    
    H_W_u = H((n+1):end,(n+1):end);
    H_W_beta = H(1:n,1:n);
    
    e = y - X*beta_theory - Z*u_theory(:,i); % USe e or e2??????
    v = Ainv*u_theory(:,i);
    se2_theory(1,i+1) = (sum(e.^2)+se2_theory(1,i)*trace(H_W_beta))/n;
    su2_theory(1,i+1) = (sum(v) + su2_theory(1,i)*trace(H_W_u))/q;

    disp([i+1,se2_theory(1,i+1),su2_theory(1,i+1)])
end
t = toc;
 

%% Lanczos

tic
[Q,R] = qr(Z); % not really needed. on can write a better QR factorization here
R = R(1:q,1:q);
Qt = Q';
Q1t = Qt(1:q,:);
Q2t = Qt((q+1):end,:);

A1 = X'*Q1t';
A2 = X'*Q2t';
C = (A1*A1'+A2*A2');
% Cinv = inv(C); % use LU
R1 = chol(C);
I2 = speye(size(C));
Cinv = R1\(R1'\I2);
     

ybar = Q'*y;
ybar1 = ybar(1:q);

ybar2 = ybar((q+1):end);
ytilda = A1*ybar1 + A2*ybar2;

%Rinv = inv(R);
Rinv =  diag( (1 ./ diag(R)).^2 ); %Not correct in general, only for the special Z


B = Rinv'*Ainv*Rinv;
r0 = (ybar1-A1'*Cinv*ytilda);
b = r0;

[Uk,T] = LAN(B,k,r0);


for i = 1:N  
    %invRHS = inv(- Uk'*A1'*Cinv*A1*Uk + eye(k) + se2_lanc(1,i)/su2_lanc(1,i)*T);
    RHS = (- Uk'*A1'*Cinv*A1*Uk + eye(k) + se2_lanc(1,i)/su2_lanc(1,i)*T);
    [L,U,P] = lu(RHS);
    I = eye(size(RHS));
    invRHS = zeros(size(RHS));
    invRHS(:,1:k) = U\(L\I(:,1:k));
    
    P0 = Rinv*Uk;
    P1 = invRHS*Uk';
    P2 = P0*P1;
    u_lanc(:,i) = P2*b;
    
    ubar = Uk*P1*b;
    beta = Cinv*(ytilda-A1*ubar);

    u_y = P2*(Q1t-A1'*Cinv*(A1*Q1t+A2*Q2t)); %L3
    TraceZu_y =  sparsetrace(Z,u_y);
    
    beta_y = Cinv*(A1*Q1t + A2*Q2t - (A1*inv(R))*u_y);
    TraceBeta_y = sparsetrace(X,beta_y);
    
    T_Hbeta = TraceZu_y + TraceBeta_y;    
    T_Hu = q - TraceZu_y; % 
    
    e = y - X*beta-Z* u_lanc(:,i); 
    v = Ainv*u_lanc(:,i);
    se2_lanc(1,i+1) = (sum(e.^2)+se2_lanc(1,i)*T_Hbeta)/n;    
    su2_lanc(1,i+1) = (sum(v) + su2_lanc(1,i)*T_Hu)/q;
    disp([i+1,se2_theory(1,i+1),su2_theory(1,i+1),se2_lanc(1,i+1),su2_lanc(1,i+1)])
    %disp([i+1,se2_lanc(1,i+1),su2_lanc(1,i+1)])
    
 end
t2 = toc;

disp([t,t2])