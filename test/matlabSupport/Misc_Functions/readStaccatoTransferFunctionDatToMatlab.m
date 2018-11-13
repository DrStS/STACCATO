function [ H ] = readStaccatoTransferFunctionDatToMatlab( path, filename, numinputs, numoutputs, numfreq )
%path: Path directory | example: 
%filename: Filename | example: 
%numinputs: Number of inputs | example: 
%numoutputs: Number of outputs | example: 
%numfreq: Number of frequency | example: 
%   Author: Harikrishnan

mat = dlmread([path,filename]);
matComplex = mat(:,1)+1i*mat(:,2);

for j = 1:numfreq
    for i = 1:numinputs
        H(i,:,j) = matComplex((i-1)*numinputs  + (j-1)*numoutputs*numoutputs +1   :(i-1)*numinputs  + (j-1)*numoutputs*numoutputs   +numoutputs );
        
    end
end

end

