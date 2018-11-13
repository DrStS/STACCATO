
H_abq = zeros(9);

numOutputs = 9;
P_n = [34 23 2];    % Node labels in order

flag= [];
for in_index = 1:numOutputs
    filename_hr = ['D:\Thesis\WorkingDirectory\MOR\KrylovSolving\AbaqusWD\wishboneRupert\Index',num2str(in_index),'\SSD_Node G12LENKER-1.'];
    for i =1:3  %node
        for j = 1:3 %dof
            [X,COMP]=importHistOut([filename_hr,num2str(P_n(i)),'_U',num2str(j),'.dat']);
            for ifreq = 1:length(X)
                H_abq(in_index, (i-1)*3+j, ifreq) = COMP(ifreq);
            end
        end
    end
    %if in_index == 3
%         for ifreq = 1:396
%             error_norm = errorCheck(H_abq(in_index, :, ifreq),system1.H_R(in_index, :, ifreq));
%             if(error_norm > 1e-5)
%                 flag = [flag error_norm]
%             end
%         end
    %end
end
err = [];
for ifreq = 1:396
    error_norm = errorCheck(H_abq(:, :, ifreq),system1.H_R(:, :, ifreq));
    err = [err error_norm]
    if(error_norm > 1e-6)
        flag = [flag error_norm]
    end
end

plot(freqs, err);
