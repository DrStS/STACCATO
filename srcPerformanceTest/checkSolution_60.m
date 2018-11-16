%% Clear
close all;
clear all;
clc;
format long;
matrix_rep = 2;
%% Read data
[K1] = mmread('r_approx_180/KSM_Stiffness_r126.mtx');
[K2] = mmread('r_approx_180/KSM_Stiffness_r132.mtx');
[K3] = mmread('r_approx_180/KSM_Stiffness_r168.mtx');
[K4] = mmread('r_approx_180/KSM_Stiffness_r174.mtx');
[K5] = mmread('r_approx_180/KSM_Stiffness_r180.mtx');
[K6] = mmread('r_approx_180/KSM_Stiffness_r186.mtx');
[K7] = mmread('r_approx_180/KSM_Stiffness_r192.mtx');
[K8] = mmread('r_approx_300/KSM_Stiffness_r288.mtx');
[K9] = mmread('r_approx_300/KSM_Stiffness_r294.mtx');
[K10] = mmread('r_approx_300/KSM_Stiffness_r300.mtx');
[K11] = mmread('r_approx_300/KSM_Stiffness_r306.mtx');
[K12] = mmread('r_approx_300/KSM_Stiffness_r312.mtx');

[M1] = mmread('r_approx_180/KSM_Mass_r126.mtx');
[M2] = mmread('r_approx_180/KSM_Mass_r132.mtx');
[M3] = mmread('r_approx_180/KSM_Mass_r168.mtx');
[M4] = mmread('r_approx_180/KSM_Mass_r174.mtx');
[M5] = mmread('r_approx_180/KSM_Mass_r180.mtx');
[M6] = mmread('r_approx_180/KSM_Mass_r186.mtx');
[M7] = mmread('r_approx_180/KSM_Mass_r192.mtx');
[M8] = mmread('r_approx_300/KSM_Mass_r288.mtx');
[M9] = mmread('r_approx_300/KSM_Mass_r294.mtx');
[M10] = mmread('r_approx_300/KSM_Mass_r300.mtx');
[M11] = mmread('r_approx_300/KSM_Mass_r306.mtx');
[M12] = mmread('r_approx_300/KSM_Mass_r312.mtx');

disp('Matrices loaded');

solCUDA = importdata('/opt/software/repos/STACCATO/test/krylovMidsizeTest/gpu/concurrent/batchedLU/output/solution.dat');
% solCUDA = importdata('/opt/software/repos/STACCATO/test/krylovMidsizeTest/cpu/sequential/sparseLU/output/solution.dat');

%solCUDA = importdata('/opt/software/repos/STACCATO/bin64/output/solution.dat');

solCUDA_complex = complex(solCUDA.data(:,1), solCUDA.data(:,2));
disp('Solution from CUDA loaded');
%% RHS
row = 2658 * matrix_rep;
real = ones(row, 1);
imag = zeros(row, 1);
b = complex(real, imag);
disp('RHS prepared');
%% Combine matrices
K = sparse(blkdiag(K1, K2, K3, K4, K5, K6, K7, K8, K9, K10, K11, K12, K1, K2, K3, K4, K5, K6, K7, K8, K9, K10, K11, K12));
M = sparse(blkdiag(M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12));
disp('Matrices combined');
%% Assemble matrix A
freq_min = 1;
freq_max = 2;
alpha = 4*pi^2;
M_tilde = alpha*M;
A_1 = K - freq_min^2*M_tilde;
A_2 = K - freq_max^2*M_tilde;
disp('Matrix assembled');
%% LU
sol_1 = A_1\b;
sol_2 = A_2\b;
disp('Matrix solved');
%% Error
sol = [sol_1; sol_2];
err = norm(sol - solCUDA_complex)/norm(sol);
fprintf('Relative error =');
disp(err);