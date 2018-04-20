function [ MAT, isComplex, IA, JA_CSR ] = readCSR( dir , prefix )
% Function to read in the CSR vectors and matrix
%   Supports STACCATO export format

format long;

%% Nomenclature
fileName_CSR_IA = [dir, prefix, '_CSR_IA.dat'];
fileName_CSR_JA = [dir, prefix, '_CSR_JA.dat'];
fileName_CSR_MAT= [dir, prefix, '_CSR_MAT.dat'];


%% IA - CSR Row
IA_CSR = load(fileName_CSR_IA);
IA_CSR = IA_CSR(:,1);

%% JA - CSR Column
JA_CSR = load(fileName_CSR_JA);
JA_CSR = JA_CSR(:,1);

%% MAT - 1 Column: Real Mat, 2 Columns: Complex Mat
MAT = load(fileName_CSR_MAT);
if(size(MAT,2) == 2)
    isComplex = true;
    type = 'Complex';
    MAT = MAT(:,1) + 1i*MAT(:,2);
elseif(size(MAT,2) == 1)
    isComplex = false;
    type = 'Real Only';
    MAT = MAT(:,1);
else
    display('Error in CSR_MAT Input.');
end

IA = zeros(size(MAT,1),1);
for i = 1:size(IA_CSR,1)-1
    numRowNonZero = IA_CSR(i+1) - IA_CSR(i);
    for j = 1:numRowNonZero
        IA(IA_CSR(i) + j - 1) = i;
    end
end

LU = sparse(IA,JA_CSR,MAT);
[n,m]=size(LU);
MAT=LU.'+LU;
MAT(1:n+1:end)=diag(LU);

X= ['--CSR Read Successful.\n-------Name: ',prefix, '_CSR_MAT.dat','\n-------Type: ', type, '\n-------Size: ',num2str(n),'x',num2str(m),'\n'];
fprintf(X);
end

