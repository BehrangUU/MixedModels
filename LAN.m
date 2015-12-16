function [U,T] = LAN(A,k,u0)
% Lanczos tridiagonalization with full reorthogonalization according to
% Stewart Matrix Alg Vol: II, p. 349

nor=norm(u0);
U=u0/nor;
beta=zeros(k-1,1); 
alpha=zeros(k,1); 
for i=1:k-1
    v=A*U(:,i);
    if i ~= 1
        v = v - beta(i-1)*U(:,i-1); 
    end
    alpha(i)=U(:,i)'*v;
    v = v - alpha(i)*U(:,i);
    for j=1:i
        v = v - (U(:,j)'*v)*U(:,j);
    end
    %v = v - (U(:,i)'*v)*U(:,i);
    beta(i)=norm(v);
    U(:,i+1)=v/beta(i);
end

T = diag(alpha)+diag(beta(1:(k-1)),1)+diag(beta(1:(k-1)),-1);