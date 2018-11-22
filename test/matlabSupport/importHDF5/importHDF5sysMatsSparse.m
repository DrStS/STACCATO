function [ K,M,S,D, map ] = importHDF5sysMatsSparse( file )
%READHDF5SYSMATRICES Function to import the sparse system matrices from
%HDF5 file container
    group='/OperatorsSparseFOM';
    
    K =readHDF5CSR(file,[group '/Kre'],1);
    S =readHDF5CSR(file,[group '/Kim'],1);
    M =readHDF5CSR(file,[group '/M'],1);
    D =readHDF5CSR(file,[group '/D'],1);
    
    %read map
    mapStr=h5read(file,[group '/nodeToDoFLabelMap']);
    map=[mapStr.nodeLabel, mapStr.DoFLabel];
end

