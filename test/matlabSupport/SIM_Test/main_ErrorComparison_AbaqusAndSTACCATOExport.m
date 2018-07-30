clear all;
clc;

addpath('.\SIM_Test\');

%% ABAQUS Solution Read
ABQ_DIR = 'D:\Thesis\WorkingDirectory\MOR\KrylovSolving\AbaqusWD\TrussTrussConnector\';
ABQ_FILE_FIELD = 'FieldOut.dat';

[ABQ_NODE_LABEL, ABQ_LOCAL_DOF, ABQ_SOLUTION] = readAbaqusFieldOut(ABQ_DIR,ABQ_FILE_FIELD);

%% STACCATO Solution Read
STACCATO_DIR = 'C:\software\repos\staccato\bin64\Release\';
STACCATO_FILE_MAP = 'MAP_UMA.dat';
STACCATO_FILE_FIELD = 'TrussTrussConnector_Solution_F100.000000.dat';

[STACCATO_SOLUTION, isComplexSOL] = readVector(STACCATO_DIR, STACCATO_FILE_FIELD);
[STACCATO_NODE_LABEL, STACCATO_LOCAL_DOF, STACCATO_GLOBAL_DOF] = readUMA_MAP(STACCATO_DIR, STACCATO_FILE_MAP);

% Accomodate STACCATO Solution to fit Abaqus
[STACCATO_SOLUTIONR] = mapStaccatoSolutionMapToAbaqusMap(STACCATO_SOLUTION, STACCATO_NODE_LABEL, STACCATO_LOCAL_DOF, STACCATO_GLOBAL_DOF, ABQ_NODE_LABEL, ABQ_LOCAL_DOF);

% Error Check
isRelative = true;
[ error_norm, error ] = errorCheck(ABQ_SOLUTION,STACCATO_SOLUTIONR, isRelative);
