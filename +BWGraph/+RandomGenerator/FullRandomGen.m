classdef FullRandomGen < BWGraph.RandomGenerator.IRandomGen
    properties (Access = private)
        MinValue;
        MaxValue;
        Type;
    end
    
    methods
        function obj = FullRandomGen(minVal, maxVal, type)
            % Конструктор принимает диапазон для случайных значений
            if nargin < 2
                error('Необходимо указать минимальное и максимальное значения');
            end
            if minVal >= maxVal
                error('Минимальное значение должно быть меньше максимального');
            end
            obj.MinValue = minVal;
            obj.MaxValue = maxVal;
            obj.Type = type;
        end
        
        function Generate(obj, graphshell)
            arguments
                obj 
                graphshell BWGraph.NonLinearBWGraph.GraphShellNonlinear
            end 
            listOfNodes = graphshell.ListOfNodes;
            
            for i = 1:numel(listOfNodes)
                currentNode = listOfNodes(i);
                currentEdges = currentNode.getOutEdges();

                for j = 1:numel(currentEdges)
                    edge = currentEdges(j);
                    switch obj.Type
                        case 'Alfa'
                            edge.Alfa = obj.MinValue + (obj.MaxValue - obj.MinValue) * rand();
                        case 'Beta'
                            edge.Beta = obj.MinValue + (obj.MaxValue - obj.MinValue) * rand();
                        case 'Gamma'
                            edge.Gamma = obj.MinValue + (obj.MaxValue - obj.MinValue) * rand();
                        case 'Delta'
                            edge.Delta = obj.MinValue + (obj.MaxValue - obj.MinValue) * rand();
                        otherwise
                            disp('Unknown type.');
                    end
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