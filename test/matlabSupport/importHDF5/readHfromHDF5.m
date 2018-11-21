function [ H ] = readHfromHDF5( fileName,freqs )
%READHFROMHDF5 Summary of this function goes here
%   Detailed explanation goes here


f=h5read(fileName,'/inputOutput/freq'); nFreq=length(f);
c=ismember(f,freqs);
inds=find(c);
if length(inds)-length(freqs)~=0
    error('interpolation would be necessary')
end

inp=h5read(fileName,'/inputOutput/inputs'); nodeLabel = inp.nLabel; inp=inp.nDof; nInp=length(inp);
A=h5read(fileName,'/inputOutput/A');
H=reshape(A.real+1j*A.imag,nInp,nInp,nFreq);
H=H(:,:,inds);

end

