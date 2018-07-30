clc;
clear all;

% Read in stiffness matrix
dir = 'C:\software\repos\staccato\bin64\Release\'
file = 'GenericSystem_stiffness.mtx';
file = 'Staccato_Sparse_Stiffness_CSR_MAT_BMW.mtx';

A = import_staccato_mtx([dir, file]);
a_full1 =full(A);
% a_full1(2,2) = 1e36;
% a_full1(3,3) = 1e36;
% a_full1(5,5) = 1e36;
% a_full1(6,6) = 1e36;
dbc = [];
for i = 1:size(a_full1,1)
    if(a_full1(i,i) >=1e36)
        dbc= [dbc i];
    end
end

file = 'GenericSystem_mass.mtx';
file = 'Staccato_Sparse_Mass_CSR_MAT.mtx';

M = import_staccato_mtx([dir, file]);
M_full2 =full(M);

%K_dyn = a_full1;
%K_dyn(4:end,4:end) = a_full1(4:end,4:end) - (2*pi*100)^2*M_full2;

[kr,kc] = size(A);
[mr,mc] = size(M);
if(kr == mr && kc == mc)
    K_dyn = A - (2*pi*100)^2*M;
elseif(kr > mr && kc > mc)
    M_full2(kr, kc) = 0;
    M(kr,kc) = 0;
    K_dyn = A - (2*pi*100)^2*M;
end

% Kill rows and columns
% for i=1:length(dbc)
%     K_dyn(:,dbc(i)) = zeros(size(K_dyn,1),1);
%     K_dyn(dbc(i),:) = zeros(1,size(K_dyn,1));
%     K_dyn(dbc(i),dbc(i)) = 1;
% end

RHS_=zeros(size(K_dyn,1),1);
RHS_(1,1) = 1;
%RHS(dbc) = ones(length(dbc),1)*1e36;

K_dyn\RHS_

[MATK, RHS, SOLUTION] = readSTACCATOExport( 'C:\software\repos\staccato\bin64\Release\','MultiEleTet' );
FULL_MAT = full(MAT);
[MATM_SIM, RHS, SOLUTION] = readSTACCATOExport( 'C:\software\repos\staccato\bin64\Release\','MultiEleTetSim' );
%norm(K_dyn - FULL_MAT)
MAT\RHS
FULL_MAT\RHS


file = 'TrussTrussDistributing_CSR_MAT.mtx';

%K_dyn_Ass = import_staccato_mtx([dir, file]);

%K_dyn_Ass - FULL_MAT;

pathabq = 'D:\Thesis\WorkingDirectory\MOR\KrylovSolving\AbaqusWD\MultiEleTet';
fileabq_st = 'MultiEleTet_STIF1.mtx';
fileabq_ms = 'MultiEleTet_MASS2.mtx';

K_1=import_abaqus_matrix([pathabq,'\', fileabq_st]);
M_1=import_abaqus_matrix([pathabq,'\', fileabq_ms]);
KDYN_ = K_1-(2*pi*250)^2*M_1;
RHSMTX = zeros(size(KDYN_,1),1);
RHSMTX(10,1) = 1;

KDYN_\RHSMTX
full(KDYN_)\RHSMTX

pathabq = 'D:\Thesis\WorkingDirectory\MOR\KrylovSolving\AbaqusWD\TrussTrussConnector';
fileabq_st = 'TrussDoubleExport_STIF1.mtx';
fileabq_ms = 'TrussDoubleExport_MASS2.mtx';

K_1=import_abaqus_matrix([pathabq,'\', fileabq_st]);
M_1=import_abaqus_matrix([pathabq,'\', fileabq_ms]);
KDYN_ = K_1-(2*pi*100)^2*M_1;
RHSMTX = zeros(size(KDYN_,1),1);
RHSMTX(1,1) = 1;
KDYN_\RHSMTX

[MAT, RHS, SOLUTION] = readSTACCATOExport( 'C:\software\repos\staccato\bin64\Release\','TrussTrussDouble' );
