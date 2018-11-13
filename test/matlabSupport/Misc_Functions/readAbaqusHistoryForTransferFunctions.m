function [ H_mat ] = readAbaqusHistoryForTransferFunctions( dir, prefix, output_nodelabel_to_dof_map, loadcasepernode, totaldof )
%This function reads in the history output series and load them to the
%transfer functions
% Author: Harikrishnan Sreekumar

% Protocol for Output
dofOutputProtocol = containers.Map('KeyType','int64','ValueType','char');
dofOutputProtocol(1) = '1';
dofOutputProtocol(2) = '2';
dofOutputProtocol(3) = '3';
dofOutputProtocol(4) = 'R1';
dofOutputProtocol(5) = 'R2';
dofOutputProtocol(6) = 'R3';

% Protocol for Input
dofInputProtocol = containers.Map('KeyType','int64','ValueType','char');
dofInputProtocol(1) = 'FX';
dofInputProtocol(2) = 'FY';
dofInputProtocol(3) = 'FZ';
dofInputProtocol(4) = 'MX';
dofInputProtocol(5) = 'MY';
dofInputProtocol(6) = 'MZ';

H_mat = [];
nodelabel = loadcasepernode; % Do not use data from map: Accessing keys would make it sorted in ascending order

for i = 1:length(nodelabel)                                     % For each node output
    for j =1:length(output_nodelabel_to_dof_map(nodelabel(i)))  % For each dof for output
        for in_index = 1:length(loadcasepernode)                % For each node of load case
            doflist = output_nodelabel_to_dof_map(nodelabel(i));
            doflistcase = output_nodelabel_to_dof_map(loadcasepernode(i));
            for in_dof_index =1:length(doflistcase)             % For each dof of load case 
                filename = [dir, prefix, num2str(nodelabel(i)),'_U',dofOutputProtocol(doflist(j)),' Load case LC',num2str(loadcasepernode(in_index)), dofInputProtocol(doflistcase(in_dof_index)),'.dat'];
                [X,COMP]=importHistOut(filename);
                for ifreq = 1:length(X)                         % Fill in for all frequencies
                    H_mat( (in_index-1)*length(doflistcase)+in_dof_index, (i-1)*length(doflist)+j, ifreq) = COMP(ifreq);
                end
            end
        end
    end
end


end

