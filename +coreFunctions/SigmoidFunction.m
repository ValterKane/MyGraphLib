classdef SigmoidFunction < coreFunctions.ICoreF
    
    methods (Static)
        function obj = createFromData(data)
            % Фабричный метод для создания экземпляра при загрузке
            obj = SigmoidFunction();
            obj.load(data);
        end
    end
    
    methods
        function obj = SigmoidFunction()
        end
        
        function  result = CalcCoreFunction(obj, InputParams)
            arguments
                obj 
                InputParams (1,1) double
            end
            result = 1 / (1 + exp(-InputParams));
        end
        
        function result = GetNumOfInputParams(obj)
            result = 1;
        end

        function data = save(obj)
            % Сериализация объекта в структуру
            data = struct();
            data.ClassName = class(obj);
        end
        
        function load(obj, data)
            % Десериализация объекта из структуры
            if ~strcmp(data.ClassName, class(obj))
                error('Class mismatch during loading');
            end
        end

    end
end

