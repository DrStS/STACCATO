function [STACCATO_SOLUTION_REDEF] = mapStaccatoSolutionMapToAbaqusMap(STACCATO_SOLUTION, STACCATO_NODE_LABEL, STACCATO_LOCAL_DOF, STACCATO_GLOBAL_DOF, ABQ_NODE_LABEL, ABQ_LOCAL_DOF)
%mapStaccatoSolutionMapToAbaqusMap Accomodates staccato solution to abaqus
%   - Solution with redefined indexing
%   - Eliminates internal DoFs

staccatoMap = containers.Map('KeyType','int64','ValueType','any');
staccatoGlobalMap = containers.Map('KeyType','int64','ValueType','any');
for i = 1:length(STACCATO_NODE_LABEL)
    if(STACCATO_NODE_LABEL(i)<1e9)
        if(staccatoMap.isKey(STACCATO_NODE_LABEL(i)))
            staccatoMap(STACCATO_NODE_LABEL(i)) = [staccatoMap(STACCATO_NODE_LABEL(i)) STACCATO_LOCAL_DOF(i)];
            staccatoGlobalMap(STACCATO_NODE_LABEL(i)) = [staccatoGlobalMap(STACCATO_NODE_LABEL(i)) STACCATO_GLOBAL_DOF(i)];
        else
            staccatoMap(STACCATO_NODE_LABEL(i)) = STACCATO_LOCAL_DOF(i);
            staccatoGlobalMap(STACCATO_NODE_LABEL(i)) = STACCATO_GLOBAL_DOF(i);
        end
    end
end

% Convert to 6 DOF
STACCATO_SOLUTION_REDEF = [];
convertTo6DOF = true;
if convertTo6DOF
    keys_st = cell2mat(staccatoMap.keys);
    for i = 1:length(keys_st)
        if size(staccatoMap(keys_st(i)),2) == 3
            STACCATO_SOLUTION_REDEF = [STACCATO_SOLUTION_REDEF ;STACCATO_SOLUTION(staccatoGlobalMap(keys_st(i))); zeros(3,1)];
        else
            STACCATO_SOLUTION_REDEF = [STACCATO_SOLUTION_REDEF ;STACCATO_SOLUTION(staccatoGlobalMap(keys_st(i)))];
        end
    end
else
    keys_st = cell2mat(staccatoGlobalMap.keys);
    for i = 1:length(keys_st)
        STACCATO_SOLUTION_REDEF = [STACCATO_SOLUTION_REDEF; STACCATO_SOLUTION(staccatoGlobalMap(keys_st(i)))]
    end
end
