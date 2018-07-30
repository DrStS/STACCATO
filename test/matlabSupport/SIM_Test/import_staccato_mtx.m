%%Importing Abaqus Matrix

function [matlab_matrix_sym] = import_staccato_mtx(mtx_file)
%============== Import Stiffness Matrix ==============%
abaqus_matrix = dlmread(mtx_file);

%matlab_stiffness_matrix=abaqus_stiffness_matrix;

% merge node number info from column 1 and DOF info from column 2 and
% store in the 1st column of a new matrix
matlab_dofs(:,1) = abaqus_matrix(:,1);

% merge node number info from column 3 and DOF info from column 4 and
% store in the 2nd column of a new matrix
matlab_dofs(:,2) = abaqus_matrix(:,2);

% extract the stiffness values from the .mtx file, and store in a double
% length vector
matrix_values = [abaqus_matrix(:,3)];
for i =1:size(abaqus_matrix,1)
    matlab_matrix(matlab_dofs(i,1),matlab_dofs(i,2)) = matrix_values(i);
end

matlab_matrix_sym = matlab_matrix + matlab_matrix';
matlab_matrix_sym(1:size(matlab_matrix_sym,1)+1:end)=diag(matlab_matrix);