classdef GraphShell
    properties
        ListOfNodes BWGraph.Node  % Вектор всех Node
    end

    methods
        function obj = GraphShell(varargin)
            obj.ListOfNodes = BWGraph.Node.empty(0, 1);
            % Добавляем узлы (переданные как аргументы)
            for i = 1:length(varargin)
                if isa(varargin{i}, 'BWGraph.Node')  % Проверяем тип
                    obj.ListOfNodes(end+1) = varargin{i};
                else
                    error('Ожидается объект класса Node.');
                end
            end
        end
        
        function Forward(obj, Data)
            arguments
                obj     BWGraph.GraphShell
                Data    BWGraph.CustomMatrix.BWMatrix
            end
            num_nods = numel(obj.ListOfNodes);
            num_of_rows = Data.rowCount();

            if num_of_rows ~= num_nods
                error('Входных значений должно быть столько же, сколько и вершин')
            end

            for i = 1:numel(obj.ListOfNodes)
                currentNode = obj.ListOfNodes(i);
                allNeighbors = currentNode.getOutEdges();
                
                G1 = 0;
                for k = 1:numel(allNeighbors)
                    G1 = G1 + (allNeighbors(k).Alfa * currentNode.getFResult() + allNeighbors(k).Beta);
                end

                G2 = 0;
                allIncomingNeighbors = obj.getIncomingNeighbors(currentNode);
                for k = 1:numel(allIncomingNeighbors)
                    edgeToCurrentNode = allIncomingNeighbors(k).getEdgeToTarget(currentNode);
                    G2 = G2 + (edgeToCurrentNode.Alfa * allIncomingNeighbors(k).getFResult() + edgeToCurrentNode.Beta);
                end

                currentData = Data.getRow(i);
                fi = currentNode.calcNodeFunc(currentData);
                currentNode.setFResult(fi + G1 - G2);
            end
        end

        function res = GetModelResults(obj)
            results = [];
            for i = 1:numel(obj.ListOfNodes)
                results(end+1) = obj.ListOfNodes(i).getFResult();
            end
            res = results;
        end
    end

    methods (Access = private)
        function incomingNodes = getIncomingNeighbors(obj, targetNode)
            % Возвращает все узлы, у которых есть ребро в targetNode
            % Вход:  targetNode - объект Node, для которого ищем входящие связи
            % Выход: incomingNodes - массив Node (вершины-источники)
            
            if ~isa(targetNode, 'BWGraph.Node')
                error('targetNode должен быть объектом класса Node.');
            end
            
            incomingNodes = BWGraph.Node.empty(0, 1);
            
            % Проходим по всем узлам графа
            for i = 1:numel(obj.ListOfNodes)
                node = obj.ListOfNodes(i);
                
                % Проверяем, есть ли у node ребро в targetNode
                if isKey(node.getOutEdgesMap(), targetNode)
                    incomingNodes(end+1) = node;
                end
            end
        end
    end

end


