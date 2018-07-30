function [ node_label, local_dof, global_dof] = readUMA_MAP( dir, file )
%readUMA_MAP Reads in the map export from STACCATO SIM Routine

map_matrix = dlmread([dir,file],' ',2,0);
node_label = map_matrix(:,1);
local_dof = map_matrix(:,2);
global_dof = map_matrix(:,3) + 1; % Convert to one based indexing

X= ['--Map Read Successful.\n-------Name: ',file,'\n'];
fprintf(X);
end

