classdef AlphaGenerator < BWGraph.RandomGenerator.IRandomGen
      
    properties (Access = private)
        Basic_Bias;
    end
    
    methods
        function obj = AlphaGenerator(Base)
            obj.Basic_Bias = Base;
        end
        
        function Generate(obj, vararg)
            arguments
                obj 
                vararg BWGraph.GraphShell
            end 
            listOfNodes = vararg.ListOfNodes;
            
            % Сначала вычислим степени всех вершин
            degrees = zeros(1, numel(listOfNodes));
            for i = 1:numel(listOfNodes)
                currentNode = listOfNodes(i);
                outDegree = numel(currentNode.getOutEdges());
                inDegree = numel(vararg.getIncomingNeighbors(currentNode));
                degrees(i) = inDegree + outDegree;
            end

            % Нормализуем степени (чтобы избежать резких перепадов)
            maxDegree = max(degrees);
            if maxDegree > 0
                normalizedDegrees = degrees / maxDegree;
            else
                normalizedDegrees = zeros(size(degrees));
            end


            for i = 1:numel(listOfNodes)
                currentNode = listOfNodes(i);
                currentEdges = currentNode.getOutEdges();

                % Получаем степени соседних вершин
                neighbors = vararg.getIncomingNeighbors(currentNode);
                neighborDegrees = arrayfun(@(n) degrees(listOfNodes == n), neighbors);

                if ~isempty(neighborDegrees)
                    meanNeighborDegree = mean(neighborDegrees);
                else
                    meanNeighborDegree = degrees(i);
                end

                % Комбинированная мера важности ребра
                edgeImportance = (degrees(i) + meanNeighborDegree) / 2;

                for j = 1:numel(currentEdges)
                    edge = currentEdges(j);
                    edge.Alfa = exp(obj.Basic_Bias * log(1 + edgeImportance)) * rand();
                end
            end


        end
    
        function data = save(obj)
            % Сериализация объекта в структуру
            data = struct();
            data.ClassName = class(obj);
            data.Basic_Bias = obj.Basic_Bias;
        end
        
        function load(obj, data)
            % Десериализация объекта из структуры
            if ~strcmp(data.ClassName, class(obj))
                error('Class mismatch during loading');
            end
            obj.Basic_Bias = data.Basic_Bias;
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

