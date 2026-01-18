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

            num_nodes = numel(obj.ListOfNodes);
            num_of_rows = Data.rowCount();

            if num_of_rows ~= num_nodes
                error('Входных значений должно быть столько же, сколько и вершин')
            end

            % Предварительно вычисляем L_v для всех узлов
            L_values = zeros(1, num_nodes);
            for i = 1:num_nodes
                currentData = Data.getRow(i);
                L_values(i) = obj.ListOfNodes(i).calcNodeFunc(currentData);
                obj.fi_result(i) = L_values(i);
            end

            % Инициализируем F_v значениями L_v (нулевое приближение)
            F_values = L_values;

            % Итеративное уточнение значений F_v (рекуррентный процесс)
            max_iterations = 100;
            tolerance = 1e-6;

            for iter = 1:max_iterations
                prev_values = F_values;

                for i = 1:num_nodes
                    currentNode = obj.ListOfNodes(i);

                    % 1. Вычисляем G_In(v) = Σ (α_e * F_u + β_e) по входящим рёбрам
                    G_In = 0;
                    incomingNeighbors = obj.getIncomingNeighbors(currentNode);

                    for k = 1:numel(incomingNeighbors)
                        neighbor = incomingNeighbors(k);
                        edgeToCurrentNode = neighbor.getEdgeToTarget(currentNode);

                        % Используем текущие значения F_u
                        neighbor_idx = find(obj.ListOfNodes == neighbor, 1);
                        F_u = F_values(neighbor_idx);

                        G_In = G_In + (edgeToCurrentNode.Alfa * F_u + edgeToCurrentNode.Beta);
                    end

                    % 2. Вычисляем Σ β_e по исходящим рёбрам
                    sum_beta_out = 0;
                    outgoingEdges = currentNode.getOutEdges();

                    for k = 1:numel(outgoingEdges)
                        edge = outgoingEdges(k);
                        sum_beta_out = sum_beta_out + edge.Beta;
                    end

                    % 3. Вычисляем Σ (α_e + 1) по исходящим рёбрам
                    sum_alpha_plus_one = 0;
                    for k = 1:numel(outgoingEdges)
                        edge = outgoingEdges(k);
                        sum_alpha_plus_one = sum_alpha_plus_one + (edge.Alfa + 1);
                    end

                    if sum_alpha_plus_one == 0
                        sum_alpha_plus_one = sum_alpha_plus_one + eps;
                    end

                    % Защита от деления на ноль
                    if abs(sum_alpha_plus_one) < eps
                        error('Знаменатель в формуле F_v близок к нулю для вершины %d', currentNode.ID);
                    end

                    % 4. Применяем формулу (2.6): F_v = (L_v + G_In(v) - Σ β_e) / Σ (α_e + 1)
                    F_values(i) = (L_values(i) + G_In - sum_beta_out) / sum_alpha_plus_one;
                end

                % Проверка сходимости
                if max(abs(F_values - prev_values)) < tolerance
                    break;
                end
            end

            % Сохраняем результаты в узлы
            for i = 1:num_nodes
                obj.ListOfNodes(i).setFResult(F_values(i));
            end

            % Сохраняем также в obj.fi_result для совместимости
            obj.fi_result = L_values;
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

        function res = GetCurrentResult(obj, XData)
            results = [];
            obj.Forward(XData);
            for i = 1:numel(obj.ListOfNodes)
                results(end+1) = obj.ListOfNodes(i).getFResult();
            end
            res = results;
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

        function flag = IsWhiteVertice(obj, index)
            wi = obj.GetWhiteNodesIndices;
            flag = ismember(wi, index);
        end

        function flag = IsBlackVertice(obj, index)
            wi = obj.GetBlackNodesIndices;
            flag = ismember(wi, index);
        end

        function SaveToFile(obj, filename)
            % Under construction
        end

             

        function derivative = computeIncomingAlphaDerivative(obj, nodeIndex, edge, visitedNodes, derivativeCache)
            % Исправленная версия согласно формуле (3.12) и исправленному Forward
            % Формула: (F_u + alpha_e * dF_u/dalpha_e) / sum(alpha_e + 1)

            % Инициализация при первом вызове
            if nargin < 4
                visitedNodes = [];
                derivativeCache = containers.Map('KeyType', 'double', 'ValueType', 'any');
            end

            % Проверка кэша
            cacheKey = nodeIndex * 1000 + edge.SourceNode.ID;
            if derivativeCache.isKey(cacheKey)
                derivative = derivativeCache(cacheKey);
                return;
            end

            % Проверка на цикл
            if ismember(nodeIndex, visitedNodes)
                derivative = NaN;
                return;
            end

            % Добавляем текущую вершину в посещённые
            visitedNodes = [visitedNodes, nodeIndex];

            currentNode = obj.ListOfNodes(nodeIndex);
            sourceNode = edge.SourceNode;
            sourceNodeIndex = find(obj.ListOfNodes == sourceNode, 1);

            % Вычисляем sum(alpha_e + 1) по исходящим рёбрам текущей вершины
            outgoingEdges = currentNode.getOutEdges();
            sum_alpha_plus_one = 0;
            for k = 1:numel(outgoingEdges)
                sum_alpha_plus_one = sum_alpha_plus_one + (outgoingEdges(k).Alfa + 1);
            end

            % Рекурсивно вычисляем dF_u/dalpha_e для вершины-источника
            dF_u_dalpha = obj.computeIncomingAlphaDerivative(sourceNodeIndex, edge, visitedNodes, derivativeCache);

            % Если производная не вычислима (например, из-за цикла), возвращаем NaN
            if isnan(dF_u_dalpha)
                derivative = NaN;
                return;
            end

            % Вычисляем F_u (значение функции в вершине-источнике)
            F_u = sourceNode.getFResult();

            % Применяем исправленную формулу
            derivative = (F_u + edge.Alfa * dF_u_dalpha) / sum_alpha_plus_one;

            % Кэшируем результат
            derivativeCache(cacheKey) = derivative;
        end
        
        function derivative = computeIncomingBetaDerivative(obj, nodeIndex, edge, visitedNodes, derivativeCache)
            % Исправленная версия согласно формуле (3.13) и исправленному Forward
            % Формула: (1 + alpha_e * dF_u/dbeta_e) / sum(alpha_e + 1)

            % Инициализация при первом вызове
            if nargin < 4
                visitedNodes = [];
                derivativeCache = containers.Map('KeyType', 'double', 'ValueType', 'any');
            end

            % Проверка кэша
            cacheKey = nodeIndex * 1000 + edge.SourceNode.ID;
            if derivativeCache.isKey(cacheKey)
                derivative = derivativeCache(cacheKey);
                return;
            end

            % Проверка на цикл
            if ismember(nodeIndex, visitedNodes)
                derivative = NaN;
                return;
            end

            % Добавляем текущую вершину в посещённые
            visitedNodes = [visitedNodes, nodeIndex];

            currentNode = obj.ListOfNodes(nodeIndex);
            sourceNode = edge.SourceNode;
            sourceNodeIndex = find(obj.ListOfNodes == sourceNode, 1);

            % Вычисляем sum(alpha_e + 1) по исходящим рёбрам текущей вершины
            outgoingEdges = currentNode.getOutEdges();
            sum_alpha_plus_one = 0;
            for k = 1:numel(outgoingEdges)
                sum_alpha_plus_one = sum_alpha_plus_one + (outgoingEdges(k).Alfa + 1);
            end

            % Рекурсивно вычисляем dF_u/dbeta_e для вершины-источника
            dF_u_dbeta = obj.computeIncomingBetaDerivative(sourceNodeIndex, edge, visitedNodes, derivativeCache);

            % Если производная не вычислима (например, из-за цикла), возвращаем NaN
            if isnan(dF_u_dbeta)
                derivative = NaN;
                return;
            end

            % Применяем исправленную формулу (1 в числителе, а не 2)
            derivative = (1 + edge.Alfa * dF_u_dbeta) / sum_alpha_plus_one;

            % Кэшируем результат
            derivativeCache(cacheKey) = derivative;
        end

        function derivative = computeOutgoingAlphaDerivative(obj, nodeIndex, edge)
            % Исправленная версия согласно формуле (3.12) и исправленному Forward
            % Формула: -F_vi / sum(alpha_e + 1)

            currentNode = obj.ListOfNodes(nodeIndex);

            % Вычисляем sum(alpha_e + 1) по исходящим рёбрам текущей вершины
            outgoingEdges = currentNode.getOutEdges();
            sum_alpha_plus_one = 0;
            for k = 1:numel(outgoingEdges)
                sum_alpha_plus_one = sum_alpha_plus_one + (outgoingEdges(k).Alfa + 1);
            end

            % F_vi (значение функции в текущей вершине)
            F_vi = currentNode.getFResult();

            % Применяем исправленную формулу
            derivative = -F_vi / sum_alpha_plus_one;
        end

        function derivative = computeOutgoingBetaDerivative(obj, nodeIndex, edge)
            % Исправленная версия согласно формуле (3.13) и исправленному Forward
            % Формула: -1 / sum(alpha_e + 1)

            currentNode = obj.ListOfNodes(nodeIndex);

            % Вычисляем sum(alpha_e + 1) по исходящим рёбрам текущей вершины
            outgoingEdges = currentNode.getOutEdges();
            sum_alpha_plus_one = 0;
            for k = 1:numel(outgoingEdges)
                sum_alpha_plus_one = sum_alpha_plus_one + (outgoingEdges(k).Alfa + 1);
            end

            % Применяем исправленную формулу
            derivative = -1 / sum_alpha_plus_one;
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


        function DrawGraph_New(obj, titleStr)
            % Метод для визуализации структуры графа с нелинейными параметрами
            % Отображает все 4 параметра на рёбрах: α, β, γ, δ

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
                        edgeLabels{end+1} = sprintf('α=%.2f\n β=%.2f', edge.Alfa, edge.Beta);
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

            % Увеличиваем размер фигуры для лучшего отображения
            set(gcf, 'Position', [100, 100, 1200, 800]);

            if ~hasEdges
                h = plot(G, ...
                    'Layout', 'force', ...
                    'NodeLabel', nodeLabels, ...
                    'MarkerSize', 16, ...
                    'NodeFontSize', 14, ...
                    'LineWidth', 2);
            else
                % Используем улучшенный layout для лучшего отображения меток
                h = plot(G, ...
                    'Layout', 'layered', ...
                    'Direction', 'down', ...
                    'NodeLabel', nodeLabels, ...
                    'ArrowSize', 15, ...
                    'LineWidth', 2, ...
                    'EdgeFontSize', 11, ...
                    'NodeFontSize', 16, ...
                    'ArrowPosition', 0.9); % Стрелки ближе к цели

                % Устанавливаем цвет для стрелок
                h.EdgeColor = [0 0 0]; % Черный цвет для всех ребер

                % Увеличиваем прозрачность для лучшей видимости меток
                h.EdgeAlpha = 0.8;
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
                    nodeColors(i,:) = [0.9 0.9 0.9]; % Белый цвет (чистый белый)
                    whiteNodes = whiteNodes + 1;
                end
            end

            % Устанавливаем свойства узлов
            h.NodeColor = [0 0 0]; % Черная окантовка для всех узлов
            h.MarkerSize = 14; % Размер узлов
            h.LineWidth = 1.5; % Толщина окантовки

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

                % Улучшаем отображение меток рёбер
                if isprop(h, 'EdgeLabelRotation')
                    h.EdgeLabelRotation = 0; % Горизонтальные метки
                end

                % Добавляем фон для меток рёбер для лучшей читаемости
                if isprop(h, 'EdgeLabelBackgroundColor')
                    h.EdgeLabelBackgroundColor = [1 1 0.9]; % Светло-желтый фон
                    h.EdgeLabelBackgroundAlpha = 0.7; % Полупрозрачный
                end
            end

            % Добавляем заголовок
            if nargin > 1 && ~isempty(titleStr)
                title(titleStr, 'FontSize', 16, 'FontWeight', 'bold');
            else
                title('Черно-белый граф с нелинейными параметрами', ...
                    'FontSize', 16, 'FontWeight', 'bold');
            end

            % Добавляем информационную панель
            infoText = sprintf('Узлов: %d (Черных: %d, Белых: %d)', ...
                numel(obj.ListOfNodes), blackNodes, whiteNodes);
            annotation('textbox', [0.02, 0.02, 0.3, 0.05], ...
                'String', infoText, ...
                'FontSize', 12, ...
                'BackgroundColor', [0.95, 0.95, 0.95], ...
                'EdgeColor', [0.5, 0.5, 0.5]);

            % Добавляем легенду с параметрами
            if hasEdges
                % Создаем панель с пояснениями параметров
                paramText = {...
                    'Параметры рёбер:', ...
                    'α - линейный коэффициент', ...
                    'β - константное смещение'};

                annotation('textbox', [0.7, 0.75, 0.2, 0.1], ...
                    'String', paramText, ...
                    'FontSize', 11, ...
                    'BackgroundColor', [0.95, 0.95, 0.95], ...
                    'EdgeColor', [0.5, 0.5, 0.5]);
            end

            % Добавляем легенду для узлов
            if blackNodes > 0 || whiteNodes > 0
                hold on;

                % Создаем фиктивные точки для легенды
                hBlack = scatter(NaN, NaN, 100, ...
                    'filled', ...
                    'MarkerFaceColor', [0 0 0], ...
                    'MarkerEdgeColor', [0 0 0], ...
                    'LineWidth', 1.5);

                hWhite = scatter(NaN, NaN, 100, ...
                    'filled', ...
                    'MarkerFaceColor', [1 1 1], ...
                    'MarkerEdgeColor', [0 0 0], ...
                    'LineWidth', 1.5);

                legendHandles = [];
                legendLabels = {};

                if blackNodes > 0
                    legendHandles = [legendHandles, hBlack];
                    legendLabels = [legendLabels, 'Черные вершины (без данных)'];
                end
                if whiteNodes > 0
                    legendHandles = [legendHandles, hWhite];
                    legendLabels = [legendLabels, 'Белые вершины (с данными)'];
                end

                % Создаем легенду
                lg = legend(legendHandles, legendLabels, ...
                    'Location', 'northeast', ...
                    'FontSize', 12);

                % Устанавливаем фон легенды
                set(lg, 'Color', [0.95, 0.95, 0.95]);

                hold off;
            end

            % Улучшаем отображение
            set(gcf, 'Color', 'w');
            axis off;
            grid off;

            % Добавляем сетку координат для лучшей ориентации
            if ~hasEdges
                grid on;
                axis on;
                xlabel('X координата');
                ylabel('Y координата');
            end

            % Автоматически подгоняем размеры для лучшего отображения
            axis tight;
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


