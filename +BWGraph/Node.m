classdef Node < handle
    properties (Access = private)
        ID (:,:) {mustBePositive}                   % Номер вершины
        OutEdgesMap                                 % Список соседних вершин
        NodeType BWGraph.NodeColor                  % Цвет вершины
        NodeFunction                                % Функции вершин
        FResult double                              % Значение в вершине
        RandomGenerator                             % Объект для генерации альфа и бета
    end
    
    methods
        function obj = Node(ID, initialValue, nodeType, nodeFunction, randomGenFunction)
            arguments
                ID {mustBePositive}
                initialValue {mustBeFinite}
                nodeType BWGraph.NodeColor
                nodeFunction coreFunctions.IBoundary
                randomGenFunction BWGraph.RandomGenerator.IRandomGen
            end

            % Инициализация объекта
            obj.ID = ID;
            obj.OutEdgesMap = dictionary(); % Инициализация dictionary без ограничений типов
            obj.NodeType = nodeType;
            obj.NodeFunction = nodeFunction;
            obj.RandomGenerator = randomGenFunction;
            obj.FResult = initialValue;

        end

        function addEdge(obj, targetNode)
             % Добавляет ребро в словарь (ключ — Target)
            arguments
                obj BWGraph.Node
                targetNode BWGraph.Node
            end
            
            if isConfigured(obj.OutEdgesMap) && obj.OutEdgesMap.isKey(targetNode)
                error('Ребро в эту вершину уже существует');
            end

            alfa = obj.RandomGenerator.Generate();
            beta = obj.RandomGenerator.Generate();

            newEdge = BWGraph.Edge(obj, targetNode, alfa, beta);
            obj.OutEdgesMap(targetNode) = newEdge;
        end

        function success = removeEdgeByTarget(obj, targetNode)
             % Удаляет ребро по Target
            arguments
                obj BWGraph.Node
                targetNode BWGraph.Node
            end
            
            if isConfigured(obj.OutEdgesMap) && obj.OutEdgesMap.isKey(targetNode)
                remove(obj.OutEdgesMap, targetNode);
                success = true;
            else
                success = false;
            end
        end

        function edge = getEdgeToTarget(obj, targetNode)
            % Возвращает ребро по Target
            arguments
                obj BWGraph.Node
                targetNode BWGraph.Node
            end
            
            if ~isConfigured(obj.OutEdgesMap) || ~obj.OutEdgesMap.isKey(targetNode)
                error('Ребро в указанную вершину не найдено');
            end
            edge = obj.OutEdgesMap(targetNode);
        end

        function neighbors = getNeighbors(obj)
            % Возвращает список соседей (Target вершин)
            if isConfigured(obj.OutEdgesMap)
                neighbors = keys(obj.OutEdgesMap);
            else
                neighbors = {};
            end
        end
        
        function edges = getOutEdges(obj)
            % Возвращает все исходящие рёбра
            if isConfigured(obj.OutEdgesMap)
                edges = values(obj.OutEdgesMap);
            else
                edges = {};
            end
        end

        function res = getFResult(obj)
            res = obj.FResult;
        end

        function setFResult(obj, value)
            arguments
                obj BWGraph.Node
                value {mustBeFinite}
            end
            obj.FResult = value;
        end

        function res = calcNodeFunc(obj, inputData)
            res = obj.NodeFunction.CalcCoreFunction(inputData);
        end

        function res = getOutEdgesMap(obj)
            res = obj.OutEdgesMap;
        end

    end
   
end