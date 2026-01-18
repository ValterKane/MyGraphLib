% Пример простой линейной функции
classdef HeatLinearRegression < coreFunctions.ICoreF
    
    methods (Static)
        function obj = createFromData(data)
            % Фабричный метод для создания экземпляра при загрузке
            obj = HeatLinearRegression();
            obj.load(data);
        end
    end
    
    methods
        function obj = HeatLinearRegression()
        
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
            result = 636.08 + 0.0016*t1 + 0.0034*t2 + 0.0061*t3 -0.3495*T1 - 0.6455*T2 + 1.2255 * T3 - 0.3547*tc;
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
