function [ mat ] = readStaccatoSeriesMtx( path, prefix, freqs, isRowMajor )
% This function reads in a series of mtx files exported from staccato
% Author: Harikrishnan Sreekumar

for i = 1:length(freqs)
    currentfile = [prefix,num2str(freqs(i)),'.000000.mtx'];
    filesize = dlmread([path,currentfile],'\t',1,0);
    filemat = dlmread([path,currentfile],'\t',2,0);
    numrow = filesize(1,1);
    numcol = filesize(1,2);
    
    % Row Major
    if isRowMajor
        for irow =1:numrow
            for icol=1:numcol
                mat(irow,icol,i) = filemat((irow-1)*numrow+icol);
            end
        end
    else % Column Major
        for irow =1:numrow
            for icol=1:numcol
                mat(icol,irow,i) = filemat((irow-1)*numrow+icol,1)+ 1i*filemat((irow-1)*numrow+icol,2);
            end
        end
    end
end

end

