classdef SimpleAddingCoreFunction < coreFunctions.ICoreF
  
    properties (Access = private)
        Num_of_x = 3;
    end

    methods
        function obj = SimpleAddingCoreFunction(num_of_x)
            obj.Num_of_x = num_of_x;
        end

        function result = CalcCoreFunction(obj, InputParams)
            arguments
                obj 
                InputParams (:,1) double {mustBeFinite} 
            end
           
            if size(InputParams,1) ~= obj.Num_of_x
                error('Количество входных данных должно быть равно количеству параметров')
            end

            result = sum(InputParams,1);
        end

        function  result = GetNumOfInputParams(obj)
            result = obj.Num_of_x;
        end

        function data = save(obj)
            % Сериализация объекта в структуру
            data = struct();
            data.ClassName = class(obj);
            data.Num_of_x = obj.Num_of_x;
        end
        
        function load(obj, data)
            % Десериализация объекта из структуры
            if ~strcmp(data.ClassName, class(obj))
                error('Class mismatch during loading');
            end
            obj.Num_of_x = data.Num_of_x;
        end
    end

    methods (Static)
        function obj = createFromData(data)
            % Фабричный метод для создания экземпляра при загрузке
            obj = SimpleAddingCoreFunction();
            obj.load(data);
        end
    end

end

