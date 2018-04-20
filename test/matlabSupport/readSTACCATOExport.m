function [ MAT, RHS, SOLUTION, isComplexMAT, isComplexRHS, isComplexSOL, rowIndices, colIndices ] = readSTACCATOExport( dir, prefix )
% Function to read in export of staccato
%   Exported CSR_IA, CSR_JA, CSR_MAT, RHS, Solution

fprintf('========= Summary: Read STACCATO-Export =========\n');
[MAT, isComplexMAT, rowIndices, colIndices] = readCSR(dir, prefix);
[RHS, isComplexRHS]                         = readVector(dir, [prefix, '_RHS.dat']);
[SOLUTION, isComplexSOL]                    = readVector(dir, [prefix, '_Solution.dat']);
fprintf('=================================================\n');
end

