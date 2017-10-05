% Read matrix
format long;
load dynStiff.txt;
i = dynStiff(:,1)+1;
j = dynStiff(:,2)+1;
v = dynStiff(:,3);
clear dynStiff;
A = sparse(i,j,v);
[n,m]=size(A);
B=A'+A;
B(1:n+1:end)=diag(A);

b = zeros(n,1);
b(1,1)=1;
b(1,2)=2;
b(1,3)=3;
c=B\b;