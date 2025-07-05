classdef (Abstract) IBoundary
    methods (Abstract)
        result = CalcCoreFunction(obj, InputParams)
        result = GetNumOfInputParams(obj);
    end
end    