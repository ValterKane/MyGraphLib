classdef GraphShell < handle

    properties( Access = private)
        AlphaGenerator         % Объект для генерации альфа
        BetaGenerator          % Объект для генерации бета
    end

    properties
        ListOfNodes BWGraph.Node  % Вектор всех Node
    end

    properties (Access = private)
        fi_result
        numOfWhiteNodes
        numOfBlackNodes
    end

    methods (Static)
        function loadedGraph = LoadFromFile(filename)
            % Статический метод для загрузки GraphShell из файла
            % Возвращает полностью инициализированный объект GraphShell
            % Вход:
            %   filename - имя файла без расширения .mat
            % Выход:
            %   loadedGraph - загруженный объект GraphShell

            % 1. Проверка существования файла
            fullFilename = [filename '.mat'];
            if ~exist(fullFilename, 'file')
                error('GraphShell:FileNotFound', 'File %s not found', fullFilename);
            end

            % 2. Загрузка данных из файла
            loadedData = load(fullFilename);
            modelData = loadedData.modelData;

            % 3. Подготовка данных для создания графа
            % 3.1. Восстановление генераторов
            alphaGen = [];
            betaGen = [];

            if isfield(modelData, 'Generators')
                % Восстановление AlphaGenerator
                if isfield(modelData.Generators, 'Alpha')
                    try
                        alphaData = modelData.Generators.Alpha;
                        alphaGen = feval([alphaData.ClassName '.createFromData'], alphaData);
                    catch e
                        warning('GraphShell:LoadAlphaGen', ...
                            'Failed to load AlphaGenerator: %s', e.message);
                    end
                end

                % Восстановление BetaGenerator
                if isfield(modelData.Generators, 'Beta')
                    try
                        betaData = modelData.Generators.Beta;
                        betaGen = feval([betaData.ClassName '.createFromData'], betaData);
                    catch e
                        warning('GraphShell:LoadBetaGen', ...
                            'Failed to load BetaGenerator: %s', e.message);
                    end
                end
            end

            % 3.2. Восстановление узлов
            nodes = BWGraph.Node.empty(0, 1);
            nodeMap = containers.Map('KeyType', 'double', 'ValueType', 'any');

            for i = 1:numel(modelData.Nodes)
                nodeInfo = modelData.Nodes(i);

                % Создание узла с временной функцией
                newNode = BWGraph.Node(...
                    nodeInfo.ID, ...
                    nodeInfo.FResult, ...
                    nodeInfo.NodeType, ...
                    []); % Функция будет загружена ниже

                % Сохраняем данные функции для последующей загрузки
                newNode.functionData = nodeInfo.FunctionData;

                nodeMap(nodeInfo.ID) = newNode;
                nodes(end+1) = newNode;
            end

            % 3.3. Восстановление ребер
            for i = 1:numel(modelData.Edges)
                edgeInfo = modelData.Edges(i);

                sourceNode = nodeMap(edgeInfo.SourceID);
                targetNode = nodeMap(edgeInfo.TargetID);

                % Создание ребра с параметрами
                newEdge = BWGraph.Edge(...
                    sourceNode, ...
                    targetNode, ...
                    edgeInfo.Alfa, ...
                    edgeInfo.Beta);

                sourceNode.addEdge(targetNode); % Добавляем связь
                edge = sourceNode.getEdgeToTarget(targetNode);
                edge.Alfa = newEdge.Alfa;       % Устанавливаем параметры
                edge.Beta = newEdge.Beta;
            end

            % 4. Создание объекта GraphShell со всеми загруженными данными
            % Преобразуем массив узлов в cell-массив для varargin
            nodeCellArray = num2cell(nodes);

            % Создаем объект GraphShell
            loadedGraph = BWGraph.GraphShell(alphaGen, betaGen, nodeCellArray{:});

            % 5. Восстановление NodeFunction для всех узлов
            for i = 1:numel(loadedGraph.ListOfNodes)
                node = loadedGraph.ListOfNodes(i);
                if ~isempty(node.functionData)
                    node.loadNodeFunction(node.functionData);
                end
                node.functionData = []; % Очищаем временные данные
            end

            % 6. Восстановление остальных свойств
            loadedGraph.fi_result = modelData.fi_result;
            loadedGraph.numOfWhiteNodes = modelData.numOfWhiteNodes;
            loadedGraph.numOfBlackNodes = modelData.numOfBlackNodes;

            % 7. Проверка целостности
            if isempty(loadedGraph.AlphaGenerator) || isempty(loadedGraph.BetaGenerator)
                warning('GraphShell:MissingGenerators', ...
                    'Some generators were not loaded properly');
            end

            disp(['GraphShell successfully loaded from ' fullFilename]);
        end

    end

    methods
        function obj = GraphShell(AlphaGenerator, BetaGenerator, varargin)

            if ~isa(AlphaGenerator, 'BWGraph.RandomGenerator.IRandomGen') & ...
                    ~isa(BetaGenerator, 'BWGraph.RandomGenerator.IRandomGen')
                error(['Генераторы для alpha и beta параметров должны' ...
                    'реализовывать интерфейс BWGraph.RandomGenerator.IRandomGen'])
            end

            obj.ListOfNodes = BWGraph.Node.empty(0, 1);
            obj.fi_result = [];
            obj.AlphaGenerator = AlphaGenerator;
            obj.BetaGenerator = BetaGenerator;


            % Проверяем, что все аргументы - объекты Node
            if ~all(cellfun(@(x) isa(x, 'BWGraph.Node'), varargin))
                error('Все аргументы должны быть объектами класса Node.');
            end

            % Собираем все ID из переданных узлов
            ids = arrayfun(@(x) x.ID, [varargin{:}]);

            % Проверяем на дубликаты ID
            if numel(ids) ~= numel(unique(ids))
                error('Обнаружены вершины с одинаковыми ID. Все ID должны быть уникальными!');
            end

            % Проверяем на пересечение с уже существующими узлами
            existing_ids = [obj.ListOfNodes.ID];
            if any(ismember(ids, existing_ids))
                error('Некоторые из переданных ID уже существуют в графе.');
            end

            % Добавляем узлы
            obj.ListOfNodes = [obj.ListOfNodes; varargin{:}];

            % Генерируем рандомные значения на рёбрах
            AlphaGenerator.Generate(obj);
            BetaGenerator.Generate(obj);

            % Обновляем счётчики узлов
            obj.numOfBlackNodes = numel(obj.GetBlackNodesIndices);
            obj.numOfWhiteNodes = numel(obj.GetWhiteNodesIndices);
        end

        function Forward(obj, Data)
            arguments
                obj     BWGraph.GraphShell
                Data    BWGraph.CustomMatrix.BWMatrix
            end
            num_nods = numel(obj.ListOfNodes);
            num_of_rows = Data.rowCount();
            f_results = zeros(1,num_nods);

            if num_of_rows ~= num_nods
                error('Входных значений должно быть столько же, сколько и вершин')
            end

            for i = 1:num_nods
                currentNode = obj.ListOfNodes(i);
                edges = currentNode.getOutEdges();

                G1 = 0;
                G2 = 0;
                G3 = 0;

                for k = 1:numel(edges)
                    edge = edges(k);
                    G1 = G1 + edge.Beta;
                    G2 = G2 + edge.Alfa;
                end

                allIncomingNeighbors = obj.getIncomingNeighbors(currentNode);

                for k = 1:numel(allIncomingNeighbors)
                    neighbor = allIncomingNeighbors(k);
                    edgeToCurrentNode = neighbor.getEdgeToTarget(currentNode);
                    G3 = G3 + (edgeToCurrentNode.Alfa * neighbor.getFResult() + edgeToCurrentNode.Beta);
                end

                currentData = Data.getRow(i);

                obj.fi_result(i) = currentNode.calcNodeFunc(currentData);

                f_results(i) = (obj.fi_result(i) + G3 - G1) / (G2+1);
                obj.ListOfNodes(i).setFResult(f_results(i));
            end

            % for i = 1:num_nods
            %     obj.ListOfNodes(i).setFResult(f_results(i));
            % end
        end

        function numOfWhiteNode = GetNumOfWhiteNode(obj)
            numOfWhiteNode = obj.numOfWhiteNodes;
        end

        function numOfBlackNode = GetNumOfBlackNode(obj)
            numOfBlackNode = obj.numOfBlackNodes;
        end

        function whiteNodes = GetWhiteNodes(obj)
            isWhiteNode = arrayfun(@(node) node.getNodeType() == BWGraph.NodeColor.White, obj.ListOfNodes);
            whiteNodes = obj.ListOfNodes(isWhiteNode);
        end

        function blackNodes = GetBlackNodes(obj)
            isBlackNode = arrayfun(@(node) node.getNodeType() == BWGraph.NodeColor.Black, obj.ListOfNodes);
            blackNodes = obj.ListOfNodes(isBlackNode);
        end

        function whiteNodeIndices = GetWhiteNodesIndices(obj)
            whiteNodeIndices = find(arrayfun(@(n) n.getNodeType() == BWGraph.NodeColor.White, obj.ListOfNodes));
        end

        function blackNodeIndices = GetBlackNodesIndices(obj)
            blackNodeIndices = find(arrayfun(@(n) n.getNodeType() == BWGraph.NodeColor.Black, obj.ListOfNodes));
        end

        function res = GetModelResults(obj)

            results = [];
            for i = 1:numel(obj.ListOfNodes)
                results(end+1) = obj.ListOfNodes(i).getFResult();
            end
            res = results;
        end

        function res = dFdAl(obj, index)
            currentNode = obj.ListOfNodes(index);
            innerNeighbor = obj.getIncomingNeighbors(currentNode);
            outerEdge = currentNode.getOutEdges();
            % G1 = 0;
            G2 = 0;
            G3 = 0;
            for j = 1:numel(outerEdge)
                edge = outerEdge(j);
                % G1 = G1 + sign(edge.Alfa);
                G2 = G2 + edge.Alfa;
                G3 = G3 + edge.Beta;
            end

            % G4 = 0;
            G5 = 0;

            for j = 1:numel(innerNeighbor)
                neighbor = innerNeighbor(j);
                edge = neighbor.getEdgeToTarget(currentNode);
                % G4 = G4 + neighbor.getFResult() * sign(edge.Alfa);
                G5 = G5 + (edge.Alfa * neighbor.getFResult() + edge.Beta);
            end

            % res = (G4 / (G2+1)) - ((G2 * (obj.fi_result(index) - G3 + G5)) /(G2+1)^2);
            res = - (obj.fi_result(index) + G5 - G3) / ((G2+1)^2);
        end

        function res = dFdBt(obj, index)
            curentNode = obj.ListOfNodes(index);
            outerEdges = curentNode.getOutEdges();
            G1 = 0;
            for i = 1:numel(outerEdges)
                edge = outerEdges(i);
                G1 = G1 + edge.Alfa;
            end

            res = -1 / (G1+1);

        end

        function res = dFdfi(obj, index)
            currentNode = obj.ListOfNodes(index);
            outerEdges = currentNode.getOutEdges();
            G1 = 0;
            for j = 1:numel(outerEdges)
                edge = outerEdges(j);
                G1 = G1 + edge.Alfa;
            end

            res = 1 / (G1 + 1);
        end

        function incomingNodes = getIncomingNeighbors(obj, targetNode)
            % Возвращает все узлы, у которых есть ребро в targetNode
            % Вход:  targetNode - объект Node, для которого ищем входящие связи
            % Выход: incomingNodes - массив Node (вершины-источники)

            if ~isa(targetNode, 'BWGraph.Node')
                error('targetNode должен быть объектом класса Node.');
            end

            incomingNodes = BWGraph.Node.empty(0, 1);

            if isempty(obj.ListOfNodes)
                return;
            end

            % Проходим по всем узлам графа
            for i = 1:numel(obj.ListOfNodes)
                node = obj.ListOfNodes(i);

                % Проверяем, есть ли у node ребро в targetNode
                oo = node.getOutEdgesMap();
                if isKey(oo, targetNode)
                    incomingNodes(end+1) = node;
                end
            end
        end

        function SaveToFile(obj, filename)
            % Under construction
        end


        function derivatives = computeAlphaDerivatives(obj, nodeIndex, visitedNodes, derivativeCache)
            % Инициализация при первом вызове
            if nargin < 3
                visitedNodes = [];
                derivativeCache = containers.Map('KeyType', 'double', 'ValueType', 'any');
            end

            % Проверка кэша
            if derivativeCache.isKey(nodeIndex)
                derivatives = derivativeCache(nodeIndex);
                return;
            end

            % Проверка на цикл
            if ismember(nodeIndex, visitedNodes)
                derivatives.outgoing = [];
                return;
            end

            % Добавляем текущую вершину в посещённые
            visitedNodes = [visitedNodes, nodeIndex];

            currentNode = obj.ListOfNodes(nodeIndex);
            outgoingEdges = currentNode.getOutEdges();
            incomingNeighbors = obj.getIncomingNeighbors(currentNode);

            % Предварительные вычисления
            alpha_out = [outgoingEdges.Alfa];
            sum_alpha_out = sum(alpha_out);
            F_vi = currentNode.getFResult();

            % Инициализация результатов
            derivatives.outgoing = struct('edge', outgoingEdges, 'derivative', num2cell(zeros(1, numel(outgoingEdges))));

            % Обработка входящих соседей
            neighborIndices = arrayfun(@(n) find(obj.ListOfNodes == n, 1), incomingNeighbors);
            neighborAlphas = arrayfun(@(n) n.getEdgeToTarget(currentNode).Alfa, incomingNeighbors);

            % Вычисляем производные для всех исходящих рёбер
            for i = 1:numel(outgoingEdges)
                edge = outgoingEdges(i);

                % Векторизованное вычисление суммы
                sum_in_terms = 0;
                for j = 1:numel(incomingNeighbors)
                    neighborDerivatives = obj.computeAlphaDerivatives(neighborIndices(j), visitedNodes, derivativeCache);
                    if ~isempty(neighborDerivatives.outgoing)
                        edgeIdx = find([neighborDerivatives.outgoing.edge] == edge, 1);
                        if ~isempty(edgeIdx) && ~isnan(neighborDerivatives.outgoing(edgeIdx).derivative)
                            sum_in_terms = sum_in_terms + neighborAlphas(j) * neighborDerivatives.outgoing(edgeIdx).derivative;
                        end
                    end
                end

                % Вычисление производной
                derivatives.outgoing(i).derivative = (-F_vi + sum_in_terms) / (1 + sum_alpha_out + alpha_out(i));
            end

            % Кэшируем результаты
            derivativeCache(nodeIndex) = derivatives;
        end

        function derivatives = computeBetaDerivatives(obj, nodeIndex, visitedNodes, derivativeCache)
            % Вычисляет производные dF/dbeta для всех рёбер, связанных с вершиной nodeIndex

            % Инициализация при первом вызове
            if nargin < 3
                visitedNodes = [];
                derivativeCache = containers.Map('KeyType', 'double', 'ValueType', 'any');
            end

            % Проверка на цикл
            if ismember(nodeIndex, visitedNodes)
                if derivativeCache.isKey(nodeIndex)
                    derivatives = derivativeCache(nodeIndex);
                    return;
                else
                    derivatives.outgoing = [];
                    return;
                end
            end

            % Добавляем текущую вершину в посещённые
            visitedNodes = [visitedNodes, nodeIndex];

            currentNode = obj.ListOfNodes(nodeIndex);
            outgoingEdges = currentNode.getOutEdges();
            incomingNeighbors = obj.getIncomingNeighbors(currentNode);

            % Сумма alpha по всем исходящим рёбрам
            sum_alpha_out = sum(arrayfun(@(e) e.Alfa, outgoingEdges));

            % Инициализация структур для результатов
            derivatives.outgoing = repmat(struct('edge', [], 'derivative', 0), 1, numel(outgoingEdges));

            % 1. Обработка исходящих рёбер (формула 3)
            for i = 1:numel(outgoingEdges)
                edge = outgoingEdges(i);

                % Сумма по входящим рёбрам: sum(alpha_k * dF_u/dbeta_e)
                sum_in_terms = 0;
                for j = 1:numel(incomingNeighbors)
                    neighbor = incomingNeighbors(j);
                    neighborIndex = find(obj.ListOfNodes == neighbor, 1);
                    neighborDerivatives = obj.computeBetaDerivatives(neighborIndex, visitedNodes, derivativeCache);

                    % Ищем производную для текущего ребра edge в соседе
                    for d = neighborDerivatives.outgoing
                        if d.edge == edge && ~isnan(d.derivative)
                            sum_in_terms = sum_in_terms + neighbor.getEdgeToTarget(currentNode).Alfa * d.derivative;
                            break;
                        end
                    end
                end

                % Вычисление производной для исходящего ребра
                derivative = (-1 + sum_in_terms) / (1 + sum_alpha_out);
                % derivative = -1  / (1 + sum_alpha_out);

                derivatives.outgoing(i).edge = edge;
                derivatives.outgoing(i).derivative = derivative;
            end

            % Кэшируем результаты
            derivativeCache(nodeIndex) = derivatives;
        end


        function [dF_dalpha, dF_dbeta] = computeAllDerivatives(obj)
            % Вычисляет все производные для всех вершин графа
            % Выход:
            %   dF_dalpha - массив структур с производными по alpha
            %   dF_dbeta - массив структур с производными по beta

            numNodes = numel(obj.ListOfNodes);
            dF_dalpha = cell(1, numNodes);
            dF_dbeta = cell(1, numNodes);

            for i = 1:numNodes
                dF_dalpha{i} = obj.computeAlphaDerivatives(i);
                dF_dbeta{i} = obj.computeBetaDerivatives(i);
            end
        end


        function DrawGraph(obj, titleStr)
            % Метод для визуализации структуры графа
            % Использует правильные свойства для изменения цвета узлов

            % Создаем пустой ориентированный граф
            G = digraph();

            % Получаем все ID узлов
            nodeIDs = [obj.ListOfNodes.ID];

            % Добавляем узлы в граф (используем строковые ID)
            for i = 1:numel(nodeIDs)
                G = addnode(G, num2str(nodeIDs(i)));
            end

            % Добавляем рёбра с метками
            edgeLabels = {};
            hasEdges = false;

            for i = 1:numel(obj.ListOfNodes)
                sourceNode = obj.ListOfNodes(i);
                edges = sourceNode.getOutEdges();

                for j = 1:numel(edges)
                    edge = edges(j);
                    targetNode = edge.TargetNode;

                    if any(nodeIDs == targetNode.ID)
                        G = addedge(G, num2str(sourceNode.ID), num2str(targetNode.ID));
                        edgeLabels{end+1} = sprintf('α=%.2f β=%.2f', edge.Alfa, edge.Beta);
                        hasEdges = true;
                    else
                        warning('Target node ID %d not found in graph', targetNode.ID);
                    end
                end
            end

            % Создаем метки узлов
            nodeLabels = arrayfun(@(x) sprintf(' v_%d', x), nodeIDs, 'UniformOutput', false);

            % Настраиваем визуализацию
            figure;

            if ~hasEdges
                h = plot(G, ...
                    'Layout', 'force', ...
                    'NodeLabel', nodeLabels, ...
                    'MarkerSize', 14, ...
                    'NodeFontSize', 12);
            else
                h = plot(G, ...
                    'Layout', 'layered', ...
                    'Direction', 'down', ...
                    'NodeLabel', nodeLabels, ...
                    'ArrowSize', 12, ...
                    'LineWidth', 1.5, ...
                    'EdgeFontSize', 12, ...
                    'NodeFontSize', 16);

                % Устанавливаем черный цвет для стрелок
                h.EdgeColor = [0 0 0]; % Черный цвет для всех ребер
            end

            % Создаем массивы цветов для узлов
            nodeColors = zeros(numel(obj.ListOfNodes), 3);
            blackNodes = 0;
            whiteNodes = 0;

            for i = 1:numel(obj.ListOfNodes)
                if obj.ListOfNodes(i).getNodeType() == BWGraph.NodeColor.Black
                    nodeColors(i,:) = [0 0 0]; % Черный цвет
                    blackNodes = blackNodes + 1;
                else
                    nodeColors(i,:) = [.5 .5 .5]; % Белый цвет
                    whiteNodes = whiteNodes + 1;
                end
            end

            % Устанавливаем цвета узлов
            h.NodeColor = [0 0 0]; % Черная окантовка для всех узлов
            h.MarkerSize = 12; % Размер узлов

            % Для MATLAB R2019b и новее можно использовать MarkerFaceColor
            if isprop(h, 'MarkerFaceColor')
                h.MarkerFaceColor = nodeColors;
            else
                % Альтернативный способ для старых версий
                for i = 1:numel(obj.ListOfNodes)
                    highlight(h, num2str(nodeIDs(i)), 'NodeColor', nodeColors(i,:));
                end
            end

            % Добавляем метки рёбер (если есть ребра)
            if hasEdges
                h.EdgeLabel = edgeLabels;
                set(h, 'EdgeLabelColor', [0 0 0]); % Черный цвет текста меток
            end

            % Добавляем заголовок
            title(titleStr, 'FontSize', 14);

            % Добавляем легенду
            legendEntries = {};
            if blackNodes > 0
                legendEntries{end+1} = 'Black Nodes';
            end
            if whiteNodes > 0
                legendEntries{end+1} = 'White Nodes';
            end

            % Добавляем легенду с размером шрифта 14 пунктов
            if blackNodes > 0 || whiteNodes > 0
                hold on;
                hBlack = scatter(NaN, NaN, 100, 'filled', 'MarkerFaceColor', [0 0 0], 'MarkerEdgeColor', [0 0 0]);
                hWhite = scatter(NaN, NaN, 100, 'filled', 'MarkerFaceColor', [.5 .5 .5], 'MarkerEdgeColor', [0 0 0]);

                legendHandles = [];
                legendLabels = {};
                if blackNodes > 0
                    legendHandles = [legendHandles, hBlack];
                    legendLabels = [legendLabels, 'Черные вершины'];
                end
                if whiteNodes > 0
                    legendHandles = [legendHandles, hWhite];
                    legendLabels = [legendLabels, 'Белые вершины'];
                end

                % Создаем легенду и устанавливаем размер шрифта
                lg = legend(legendHandles, legendLabels, 'Location', 'best');
                set(lg, 'FontSize', 14);  % Устанавливаем размер шрифта 14 пунктов
                hold off;
            end

            % Улучшаем отображение
            set(gcf, 'Color', 'w');
            axis off;
            grid off;
        end

        function incomingEdges = getIncomingEdges(obj, targetNode)
            % Возвращает все входящие ребра для указанного узла
            % Вход:
            %   targetNode - объект Node, для которого ищем входящие ребра
            % Выход:
            %   incomingEdges - массив объектов Edge (входящие ребра)

            if ~isa(targetNode, 'BWGraph.Node')
                error('targetNode должен быть объектом класса Node.');
            end

            incomingEdges = BWGraph.Edge.empty(0, 1); % Инициализация пустого массива ребер

            % Проходим по всем узлам графа
            for i = 1:numel(obj.ListOfNodes)
                currentNode = obj.ListOfNodes(i);

                % Получаем все исходящие ребра текущего узла
                outEdges = currentNode.getOutEdges();

                % Проверяем каждое ребро на соответствие целевому узлу
                for j = 1:numel(outEdges)
                    edge = outEdges(j);
                    if edge.TargetNode == targetNode
                        incomingEdges(end+1) = edge;
                    end
                end
            end
        end

    end

end


