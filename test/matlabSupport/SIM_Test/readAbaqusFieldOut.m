function [ node_label, local_dof, solution] = readAbaqusFieldOut( dir, file )
%readAbaqusFieldOut Reads in the field output export from abaqus with the
%python script

map_matrix = dlmread([dir,file],' ',2,0);
node_label = map_matrix(:,1);
local_dof = map_matrix(:,2);

if size(map_matrix, 2) == 5
    type = 'Complex';
elseif size(map_matrix, 2) == 4
    type = 'Real';
else
    type = 'Unknown';
end
solution = map_matrix(:,3) + 1i*map_matrix(:,4);

X= ['--Vector Read Successful.\n-------Name: ',file,'\n-------Type: ', type, '\n-------Size: ',num2str(size(solution)),'\n'];
fprintf(X);
end

