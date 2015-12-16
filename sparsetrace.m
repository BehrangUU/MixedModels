function out = sparsetrace(U1,U2)

[n,~] = size(U1);
out = 0;
for i=1:n
   out = out + U1(i,:)*U2(:,i);
end