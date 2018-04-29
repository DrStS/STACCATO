function [ ] = verifyStaccatoExports( dir, prefix )
% Verifies the staccato export files
%   Reads in all exports -> Perform A\RHS -> Perform Error Check with
%   StaccatoResults
format long;
[A, RHS, ExpectedSol] = readSTACCATOExport(dir, prefix);
NumericalSol = A\RHS;
errorCheck(ExpectedSol, NumericalSol);

end

