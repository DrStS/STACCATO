% Read matrix
close all;
clc;
format long;
load bin64/Release/dynStiff.dat;
i = dynStiff(:,1)+1;
j = dynStiff(:,2)+1;
v = dynStiff(:,3);
clear dynStiff;
A = sparse(i,j,v);
[n,m]=size(A);
B=A'+A;
B(1:n+1:end)=diag(A);


b = zeros(n,1);
b(1,1)=1.;
b(2,1)=2.;
b(3,1)=3.;
%c=B\b;
p = symrcm(B);
R = B(p,p);
r = b(p,1);
spy(B)
figure()
spy(R);
tic
c=R\r;
toc
tic
tol = 1e-6;
maxit = 40;
[L,U] = ilu(R,struct('type','ilutp','droptol',1e-5));
%[ci,fl0,rr0,it0,rv0] = bicgstabl(R,r,tol,maxit,L,U);
[ci,fl0,rr0,it0,rv0] = gmres(R,r,[],tol,maxit,L,U);
semilogy(rv0/norm(r),'-o');
xlabel('Iteration number');
ylabel('Relative residual');
toc
%find(p<4,3)
c(35622)-ci(35622)
c(35621)-ci(35621)
c(35620)-ci(35620)
