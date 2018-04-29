function [ vec, isComplex ] = readVector( dir, fileName )
%Read in the dat file

format long;
vec = load([dir, fileName]);
if(size(vec,2) == 2)
    isComplex = true;
    type = 'Complex';
    vec = vec(:,1) + 1i*vec(:,2);
elseif(size(vec,2) == 1)
    isComplex = false;
    type = 'Real Only';
    vec = vec(:,1);
else
    display('Error in CSR_MAT Input.');
end

X= ['--Vector Read Successful.\n-------Name: ',fileName,'\n-------Type: ', type, '\n-------Size: ',num2str(size(vec)),'\n'];
fprintf(X);
end

