function [ MAT, IA, JA_CSR ] = readHDF5CSR( fName , group, issymmetric )
% Function to read in the CSR vectors and matrix
%   Supports STACCATO export format

%try
    % import from HDF5

    IA_CSR=h5read(fName,[group '/iIndices']);
    JA_CSR=h5read(fName,[group '/jIndices']);
    MAT=h5read(fName,[group '/values']);

    % MAT - 1 Column: Real Mat, 2 Columns: Complex Mat

    % if(size(MAT,2) == 2)
    %     isComplex = true;
    %     MAT = MAT(:,1) + 1i*MAT(:,2);
    % elseif(size(MAT,2) == 1)
    %     isComplex = false;
    %     MAT = MAT(:,1);
    % else
    %     display('Error in CSR_MAT Input.');
    % end

    IA = zeros(size(MAT,1),1);
    clicker = [];
    for i = 1:size(IA_CSR,1)-1
        numRowNonZero = IA_CSR(i+1) - IA_CSR(i);
        for j = 1:numRowNonZero
            IA(IA_CSR(i) + j - 1) = i;
        end
        if numRowNonZero == 0
            clicker = [clicker i];
        end
    end

    LU = sparse(IA,double(JA_CSR),MAT);
    for i=1:length(clicker)
        LU(clicker(i),clicker(i)) = 0;
    end

    [n,m]=size(LU);
    if(issymmetric)
        MAT=LU.'+LU;
        MAT(1:n+1:end)=diag(LU);
    else
        MAT = LU;
    end
    
% catch
%      MAT=[];
%      IA =[];
%      JA_CSR=[];
% end

end

