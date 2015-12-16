%#------------------------------------------------------------------#
%# Function   :  Fitting linear mixed model - Case 2                #
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
ASqrt = R\(R'\I2);
     
sigma_e = sqrt(6.2); 
sigma_u = sqrt(1);

%Z = eye(q);
Z = (kron(eye(q),ones(n/q,1)));

MU = zeros(q,1)';
SIGMA = sigma_u^2;
V = mvnrnd(MU,SIGMA*eye(q));
u = V';
a = chol(A)'*u;

e = mvnrnd(zeros(n,1)',eye(n)*sigma_e^2)';
y = Z*a + e;

%% Initilization - part 2
sigma_e = sqrt(2); 
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
%DIRECT
tic;
%zz= sparse(Z'*Z); %for using sparse cholesky
for i=1:N  
    
    LHS = Z'*Z + se2_theory(1,i)/su2_theory(1,i)*Ainv;
    %invLHS = inv(LHS);
    
%    [L,U,P] = lu(LHS);
 %   I=eye(size(LHS));
  %  invLHS=zeros(size(LHS));
   % invLHS(:,1:q)=U\(L\I(:,1:q));
   
     R = chol(LHS);
     I = eye(size(LHS));
     invLHS = R\(R'\I);

    u_theory(:,i) = invLHS*Z'*y; 
    
    L3 = invLHS*Z';
    H_u = eye(q)-AinvSqrt*invLHS*L3*Z*ASqrt';
    H_beta = Z*L3;
    
    e = y - Z*u_theory(:,i); 
    v = Ainv*u_theory(:,i);
    se2_theory(1,i+1) = (sum(e.^2)+se2_theory(1,i)*trace(H_beta))/n;
    su2_theory(1,i+1) = (sum(v) + su2_theory(1,i)*trace(H_u))/q;
   
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

ybar = Q'*y;
ybar1 = ybar(1:q);

%Rinv = inv(R);% this should be avoided
Rinv =  diag( (1 ./ diag(R)).^2 ); % Not correct in general, only for the special Z

B = sparse(Rinv'*Ainv*Rinv); %This should be avoided. 
r0 = ybar1;

[Uk,T] = LAN(B,k,r0);

yDtilda = Uk'*ybar1;

for i=1:N  
    %invRHS = inv(eye(k) + se2_lanc(1,i)/su2_lanc(1,i)*T);
    LHS = eye(k) + se2_lanc(1,i)/su2_lanc(1,i)*T;
    [L,U,P] = lu(LHS);
    I = eye(size(LHS));
    invLHS = zeros(size(LHS));
    invLHS(:,1:k) = U\(L\I(:,1:k));
    
    P0 = Rinv*Uk;
    P1 = P0*invLHS;
    P2 = yDtilda;
    u_lanc(:,i) = P1*P2;
    P3 = Uk'*Q1t;   

    T_Hbeta = sparsetrace(Z*P1,P3);% trace of H Beta
    T_Hu = q - T_Hbeta;
    
    e = y - Z*u_lanc(:,i); 
   
    se2_lanc(1,i+1) = (sum(e.^2)+se2_lanc(1,i)*T_Hbeta)/n;    
    su2_lanc(1,i+1) = (u_lanc(:,i)'*Ainv*u_lanc(:,i) + su2_lanc(1,i)*T_Hu)/q;
    
    disp([i+1,se2_theory(1,i+1),su2_theory(1,i+1),se2_lanc(1,i+1),su2_lanc(1,i+1)])
end
t2 = toc;

disp([t,t2])

