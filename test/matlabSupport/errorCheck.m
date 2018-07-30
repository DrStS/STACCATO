function [ error_norm, error ] = errorCheck( expectedSol, numericalSol, isRelative )
% Verify the expected and computed results
format long;
type = '';
if isRelative
    error = (expectedSol - numericalSol)./expectedSol;
    type = 'Relative';
else
    error = (expectedSol - numericalSol);
    type = 'Absolute';
end
error_norm = norm(error, 2);

if(error_norm > 1e-6)
    X = ['Computation Failed with ',type,' Error Norm: ', num2str(error_norm),'\n'];
    fprintf(X);
else
    X = ['Computation Successfull with ',type,' Error Norm: ', num2str(error_norm),'\n'];
    fprintf(X);
end

end

