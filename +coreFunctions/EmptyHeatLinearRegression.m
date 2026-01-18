% Пример простой линейной функции
classdef EmptyHeatLinearRegression < coreFunctions.ICoreF
    
    methods (Static)
        function obj = createFromData(data)
            % Фабричный метод для создания экземпляра при загрузке
            obj = HeatLinearRegression();
            obj.load(data);
        end
    end
    
    methods
        function obj = EmptyHeatLinearRegression()
        
        end
        
        function  result = CalcCoreFunction(obj, InputParams)
            arguments
                obj 
                InputParams (7,1) double
            end
            t1 = InputParams(1,1);
            t2 = InputParams(2,1);
            t3 = InputParams(3,1);
            T1 = InputParams(4,1);
            T2 = InputParams(5,1);
            T3 = InputParams(6,1);
            tc = InputParams(7,1);
            result = t1 + t2 + t3 - T1 - T2 + T3 - tc;
        end
        
        function result = GetNumOfInputParams(obj)
            result = 7;
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
