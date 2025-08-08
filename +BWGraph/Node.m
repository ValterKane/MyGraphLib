classdef Node < handle
    properties (Access = private) 
        OutEdgesMap                                 % Список соседних вершин
        NodeType BWGraph.NodeColor                  % Цвет вершины
        NodeFunction                                % Функции вершин
        FResult double                              % Значение в вершине
        
    end

    properties (Access = public)
         ID (:,:) {mustBePositive}                   % Номер вершины
    end
    
    methods
        function obj = Node(ID, initialValue, nodeType, nodeFunction)
            arguments
                ID {mustBePositive}
                initialValue {mustBeFinite}
                nodeType BWGraph.NodeColor
                nodeFunction
            end
            
            % Проверка: nodeFunction может быть пустым только для White узлов
            if isempty(nodeFunction) && (nodeType ~= BWGraph.NodeColor.Black)
                error('NodeFunction может быть пустой только для черных вершин!');
            end

            % Проверка типа, если nodeFunction не пустой
            if ~isempty(nodeFunction) && ~isa(nodeFunction, 'coreFunctions.ICoreF')
                error('NodeFunction должен реализовывать интерфейс coreFunctions.ICoreF.');
            end

            % Инициализация объекта
            obj.ID = ID;
            obj.OutEdgesMap = dictionary(BWGraph.Node.empty, BWGraph.Edge.empty); % Инициализация dictionary без ограничений типов
            obj.NodeType = nodeType;
            obj.NodeFunction = nodeFunction;
            obj.FResult = initialValue;
        end

        function res = getNodeType(obj)
            res = obj.NodeType;
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

            newEdge = BWGraph.Edge(obj, targetNode, 1, 1);
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
            if isempty(obj.NodeFunction)
                res = 0;
            else
                res = obj.NodeFunction.CalcCoreFunction(inputData);
            end
        end

        function res = getOutEdgesMap(obj)
            res = obj.OutEdgesMap;
        end

        function func = getNodeFunction(obj)
            func = obj.NodeFunction;
        end
        
    end
   
end