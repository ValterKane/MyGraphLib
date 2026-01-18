classdef FullRandomBetaGen < BWGraph.RandomGenerator.IRandomGen
    properties (Access = private)
        MinValue;
        MaxValue;
    end
    
    methods
        function obj = FullRandomBetaGen(minVal, maxVal)
            % Конструктор принимает диапазон для случайных значений
            if nargin < 2
                error('Необходимо указать минимальное и максимальное значения');
            end
            if minVal >= maxVal
                error('Минимальное значение должно быть меньше максимального');
            end
            obj.MinValue = minVal;
            obj.MaxValue = maxVal;
        end
        
        function Generate(obj, vararg)
            arguments
                obj 
                vararg BWGraph.GraphShell
            end 
            listOfNodes = vararg.ListOfNodes;
            
            for i = 1:numel(listOfNodes)
                currentNode = listOfNodes(i);
                currentEdges = currentNode.getOutEdges();

                for j = 1:numel(currentEdges)
                    edge = currentEdges(j);
                    % Генерируем случайное значение в заданном диапазоне
                    edge.Beta = obj.MinValue + (obj.MaxValue - obj.MinValue) * rand();
                end
            end
        end
    
        function data = save(obj)
            % Сериализация объекта в структуру
            data = struct();
            data.ClassName = class(obj);
            data.MinValue = obj.MinValue;
            data.MaxValue = obj.MaxValue;
        end
        
        function load(obj, data)
            % Десериализация объекта из структуры
            if ~strcmp(data.ClassName, class(obj))
                error('Class mismatch during loading');
            end
            obj.MinValue = data.MinValue;
            obj.MaxValue = data.MaxValue;
        end
    end

     methods (Static)
        function obj = createFromData(data)
            % Фабричный метод для создания экземпляра при загрузке
            obj = BetaGenerator(data.MinValue, data.MaxValue);
        end
    end
end