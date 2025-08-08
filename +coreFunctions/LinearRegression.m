classdef LinearRegression < coreFunctions.ICoreF
    
    properties (Access = private)
        Num_of_Params = 0;
    end

    methods (Static)
        function obj = createFromData(data)
            % Фабричный метод для создания экземпляра при загрузке
            obj = LinearRegression();
            obj.load(data);
        end
    end
    
    methods
        function obj = LinearRegression(num_of_entries)
            obj.Num_of_Params = num_of_entries;
        end
        
        function  result = CalcCoreFunction(obj, InputParams)
            arguments
                obj 
                InputParams (1,:) double
            end
            
            result = sum(InputParams);
        end
        
        function result = GetNumOfInputParams(obj)
            result = obj.Num_of_Params;
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

