function [ error_norm ] = errorCheck( expectedSol, numericalSol )
% Verify the expected and computed results
format long;
error = expectedSol - numericalSol;
error_norm = norm(error, 2);

if(error_norm > 1e-6)
    X = ['Computation Failed with Error Norm: ', num2str(error_norm),'\n'];
    fprintf(X);
else
    X = ['Computation Successfull with Error Norm: ', num2str(error_norm),'\n'];
    fprintf(X);
end

end

