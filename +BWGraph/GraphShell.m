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
        function obj = GraphShell(AlphaGenerator, BetaGenerator, NodeWeight, varargin)

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

            % Устанавливаем гамма-параметры
            gammas = num2cell(NodeWeight);
            [obj.ListOfNodes.Gamma] = gammas{:};
        end


        % Прямой матричный подход
        function Forward(obj, Data)
            arguments
                obj     BWGraph.GraphShell
                Data    BWGraph.CustomMatrix.BWMatrix
            end

            num_nodes = numel(obj.ListOfNodes);

            if Data.rowCount() ~= num_nodes
                error('Размерность данных не соответствует числу вершин');
            end

            % 1. Строим матрицы
            [D, A_in, B_in, B_out, L] = obj.buildSystemMatrices(Data);

            % 2. Система: (D - A_in)·Φ = L + B_in - B_out
            M = D - A_in;
            R = L + B_in - B_out;

            % % 3. Проверка условия (2.10)
            % D_inv = diag(1./diag(D));
            % if max(sum(abs(D_inv * A_in), 2)) >= 1
            %     warning('Модель может быть неустойчивой');
            % end

            % 4. Решение
            try
                F_vector = M \ R;
            catch
                F_vector = pinv(M) * R;
            end

            % 5. Сохранение
            for i = 1:num_nodes
                obj.ListOfNodes(i).setFResult(F_vector(i));
            end
            obj.fi_result = F_vector;
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

     
        function [D, A_in, B_in, B_out, L] = buildSystemMatrices(obj, Data)
            % Возвращает гарантированно диагональную матрицу D

            num_nodes = numel(obj.ListOfNodes);

            % ВАЖНО: D должна быть матрицей n x n, а не вектором!
            D = zeros(num_nodes, num_nodes);  % Матрица
            A_in = zeros(num_nodes, num_nodes);
            B_in = zeros(num_nodes, 1);
            B_out = zeros(num_nodes, 1);
            L = zeros(num_nodes, 1);

            for i = 1:num_nodes
                currentNode = obj.ListOfNodes(i);

                % 1. Вычисляем L(i)
                if nargin > 1 && ~isempty(Data)
                    currentData = Data.getRow(i);
                    L(i) = currentNode.calcNodeFunc(currentData);
                else
                    L(i) = currentNode.getFResult();  % Или 0
                end

                % 2. Диагональный элемент D(i,i) - и только он!
                outgoingEdges = currentNode.getOutEdges();
                sum_alpha_plus_one = 0;

                for k = 1:numel(outgoingEdges)
                    edge = outgoingEdges(k);
                    sum_alpha_plus_one = sum_alpha_plus_one + (edge.Alfa + 1);
                end

                % ЗАПИСЫВАЕМ ТОЛЬКО В ДИАГОНАЛЬНЫЙ ЭЛЕМЕНТ
                D(i, i) = sum_alpha_plus_one;

                if D(i, i) == 0
                    error(['Знаменатель равен нулю для вершины ', num2str(i)]);
                end

                % 3. B_out(i)
                for k = 1:numel(outgoingEdges)
                    edge = outgoingEdges(k);
                    B_out(i) = B_out(i) + edge.Beta;
                end

                % 4. A_in и B_in (входящие ребра)
                incomingEdges = obj.getIncomingEdges(currentNode);
                for k = 1:numel(incomingEdges)
                    edge = incomingEdges(k);
                    sourceNode = edge.SourceNode;
                    source_idx = find(obj.ListOfNodes == sourceNode, 1);

                    if ~isempty(source_idx)
                        A_in(i, source_idx) = edge.Alfa;
                        B_in(i) = B_in(i) + edge.Beta;
                    end
                end
            end
        end

        function dF_dGamma = computeGammaDerivativeForNode(obj, nodeIndex, inputData)
            
            % Текущая вершина           
            v = obj.ListOfNodes(nodeIndex);

            % Значение выхода по ядру вершины
            L_v = v.calcRawCoreFunction(inputData);

            % Выходные ребра вершины
            outgoingEdges = v.getOutEdges();

            denominator = 0;
            for k = 1:numel(outgoingEdges)
                denominator = denominator + (outgoingEdges(k).Alfa + 1);
            end

            if abs(denominator) < eps
                error(['Знаменатель близок к нулю для вершины ', num2str(nodeIndex)]);
            end

            dF_dGamma = L_v / denominator;
        end

        function dF_dalpha = computeOutgoingAlphaDerivativeForEdge(obj, nodeIndex)
            % Вспомогательная функция для прямого вычисления
            % ∂F_v/∂α_e для ИСХОДЯЩЕГО ребра без проверки кэша

            F_v = obj.ListOfNodes(nodeIndex).getFResult();

            currentNode = obj.ListOfNodes(nodeIndex);
            outgoingEdges = currentNode.getOutEdges();

            denominator = 0;
            for k = 1:numel(outgoingEdges)
                denominator = denominator + (outgoingEdges(k).Alfa + 1);
            end

            if abs(denominator) < eps
                error(['Знаменатель близок к нулю для вершины ', num2str(nodeIndex)]);
            end

            dF_dalpha = -F_v / denominator;
        end

        function dF_dbeta = computeOutgoingBetaDerivativeForEdge(obj, nodeIndex)
            % Вспомогательная функция для прямого вычисления
            % ∂F_v/∂β_e для ИСХОДЯЩЕГО ребра без проверки кэша

            currentNode = obj.ListOfNodes(nodeIndex);
            outgoingEdges = currentNode.getOutEdges();

            denominator = 0;
            for k = 1:numel(outgoingEdges)
                denominator = denominator + (outgoingEdges(k).Alfa + 1);
            end

            if abs(denominator) < eps
                error(['Знаменатель близок к нулю для вершины ', num2str(nodeIndex)]);
            end

            dF_dbeta = -1 / denominator;
        end

        function [alpha_derivatives, beta_derivatives, gama_derivatives] = computeAllDerivativesInOrder(obj, XData)
            % Вычисляет все производные в топологическом порядке
            % Возвращает:
            %   alpha_derivatives - Map: 'node_edge' → ∂F/∂α
            %   beta_derivatives  - Map: 'node_edge' → ∂F/∂β

            % 1. Получаем топологический порядок вершин
            topo_order = obj.getTopologicalOrder();

            % Инициализируем результаты
            alpha_derivatives = containers.Map;
            beta_derivatives = containers.Map;
            gama_derivatives = containers.Map;

            % 2. Вычисляем в топологическом порядке
            for i = 1:numel(topo_order)
                nodeIdx = topo_order(i);

                % Определеяем производную по Gamma
                key_gamma = sprintf('node%d_gamma', nodeIdx);
                currentData = XData.getRow(nodeIdx);
                dF_dgamma = computeGammaDerivativeForNode(obj,nodeIdx,currentData);
                gama_derivatives(key_gamma) = dF_dgamma;

                currentNode = obj.ListOfNodes(nodeIdx);
                % A. Сначала исходящие ребра (не зависят от других производных)
                outgoingEdges = currentNode.getOutEdges();
                for k = 1:numel(outgoingEdges)
                    edge = outgoingEdges(k);

                    % Исходящая производная по α
                    key_alpha = sprintf('node%d_edge%d_alpha_out', nodeIdx, edge.ID);
                    dF_dalpha = computeOutgoingAlphaDerivativeDirect(obj, nodeIdx);
                    alpha_derivatives(key_alpha) = dF_dalpha;

                    % Исходящая производная по β
                    key_beta = sprintf('node%d_edge%d_beta_out', nodeIdx, edge.ID);
                    dF_dbeta = computeOutgoingBetaDerivativeDirect(obj, nodeIdx);
                    beta_derivatives(key_beta) = dF_dbeta;
                end

                % B. Теперь входящие ребра (могут использовать уже вычисленные исходящие)
                incomingEdges = obj.getIncomingEdges(currentNode);
                for k = 1:numel(incomingEdges)
                    edge = incomingEdges(k);
                    sourceNode = edge.SourceNode;
                    sourceIdx = find(obj.ListOfNodes == sourceNode, 1);

                    if isempty(sourceIdx)
                        continue;
                    end

                    % Находим соответствующее исходящее ребро из source
                    sourceNodeObj = obj.ListOfNodes(sourceIdx);
                    sourceOutEdges = sourceNodeObj.getOutEdges();
                    dF_source_dalpha = 0;
                    dF_source_dbeta = 0;

                    for m = 1:numel(sourceOutEdges)
                        sourceEdge = sourceOutEdges(m);
                        if sourceEdge.TargetNode == currentNode
                            % Нашли соответствующее ребро
                            key_source_alpha = sprintf('node%d_edge%d_alpha_out', sourceIdx, sourceEdge.ID);
                            key_source_beta = sprintf('node%d_edge%d_beta_out', sourceIdx, sourceEdge.ID);

                            if isKey(alpha_derivatives, key_source_alpha)
                                dF_source_dalpha = alpha_derivatives(key_source_alpha);
                            end
                            if isKey(beta_derivatives, key_source_beta)
                                dF_source_dbeta = beta_derivatives(key_source_beta);
                            end
                            break;
                        end
                    end

                    % Входящая производная по α
                    key_alpha = sprintf('node%d_edge%d_alpha_in', nodeIdx, edge.ID);
                    dF_dalpha = computeIncomingAlphaDerivativeDirect(obj, nodeIdx, edge, dF_source_dalpha);
                    alpha_derivatives(key_alpha) = dF_dalpha;

                    % Входящая производная по β
                    key_beta = sprintf('node%d_edge%d_beta_in', nodeIdx, edge.ID);
                    dF_dbeta = computeIncomingBetaDerivativeDirect(obj, nodeIdx, edge, dF_source_dbeta);
                    beta_derivatives(key_beta) = dF_dbeta;
                end
            end
        end

        function dF_dalpha = computeOutgoingAlphaDerivativeDirect(obj, nodeIndex)
            % Прямое вычисление ∂F_v/∂α для ИСХОДЯЩЕГО ребра
            % Не зависит от других производных

            F_v = obj.ListOfNodes(nodeIndex).getFResult();

            currentNode = obj.ListOfNodes(nodeIndex);
            outgoingEdges = currentNode.getOutEdges();

            denominator = 0;
            for k = 1:numel(outgoingEdges)
                denominator = denominator + (outgoingEdges(k).Alfa + 1);
            end

            if abs(denominator) < eps
                error(['Знаменатель близок к нулю для вершины ', num2str(nodeIndex)]);
            end

            dF_dalpha = -F_v / denominator;
        end

        function dF_dbeta = computeOutgoingBetaDerivativeDirect(obj, nodeIndex)
            % Прямое вычисление ∂F_v/∂β для ИСХОДЯЩЕГО ребра
            % Не зависит от других производных

            currentNode = obj.ListOfNodes(nodeIndex);
            outgoingEdges = currentNode.getOutEdges();

            denominator = 0;
            for k = 1:numel(outgoingEdges)
                denominator = denominator + (outgoingEdges(k).Alfa + 1);
            end

            if abs(denominator) < eps
                error(['Знаменатель близок к нулю для вершины ', num2str(nodeIndex)]);
            end

            dF_dbeta = -1 / denominator;
        end

        function dF_dalpha = computeIncomingAlphaDerivativeDirect(obj, nodeIndex, edge, dF_source_dalpha)
            % Прямое вычисление ∂F_v/∂α для ВХОДЯЩЕГО ребра
            % Принимает уже вычисленную ∂F_u/∂α как параметр

            sourceNode = edge.SourceNode;
            sourceIdx = find(obj.ListOfNodes == sourceNode, 1);

            if isempty(sourceIdx)
                error('Исходная вершина не найдена');
            end

            F_u = obj.ListOfNodes(sourceIdx).getFResult();
            alpha_e = edge.Alfa;

            % Знаменатель для вершины v
            currentNode = obj.ListOfNodes(nodeIndex);
            outgoingEdges = currentNode.getOutEdges();

            denominator = 0;
            for k = 1:numel(outgoingEdges)
                denominator = denominator + (outgoingEdges(k).Alfa + 1);
            end

            if abs(denominator) < eps
                error(['Знаменатель близок к нулю для вершины ', num2str(nodeIndex)]);
            end

            % Формула (3.13)
            dF_dalpha = (F_u + alpha_e * dF_source_dalpha) / denominator;
        end

        function dF_dbeta = computeIncomingBetaDerivativeDirect(obj, nodeIndex, edge, dF_source_dbeta)
            % Прямое вычисление ∂F_v/∂β для ВХОДЯЩЕГО ребра
            % Принимает уже вычисленную ∂F_u/∂β как параметр

            alpha_e = edge.Alfa;

            % Знаменатель для вершины v
            currentNode = obj.ListOfNodes(nodeIndex);
            outgoingEdges = currentNode.getOutEdges();

            denominator = 0;
            for k = 1:numel(outgoingEdges)
                denominator = denominator + (outgoingEdges(k).Alfa + 1);
            end

            if abs(denominator) < eps
                error(['Знаменатель близок к нулю для вершины ', num2str(nodeIndex)]);
            end

            % Формула (3.14)
            dF_dbeta = (2 + alpha_e * dF_source_dbeta) / denominator;
        end

        function DrawGraph_New(obj, titleStr)
            % Метод для визуализации структуры графа с нелинейными параметрами
            % Отображает все 4 параметра: α, β, γ, δ

            % Создаем пустой ориентированный граф
            G = digraph();

            % Получаем все ID узлов
            nodeIDs = [obj.ListOfNodes.ID];
            nodeGammas = [obj.ListOfNodes.Gamma];

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
            nodeLabels = arrayfun(@(x,g) sprintf(' v_%d γ=%.2f', x, g), nodeIDs,nodeGammas, 'UniformOutput', false);

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

    methods(Access = private)
        function topologicalOrder = getTopologicalOrder(obj)
            % Возвращает индексы вершин в топологическом порядке
            % Используется для правильного вычисления производных

            num_nodes = numel(obj.ListOfNodes);
            visited = false(1, num_nodes);
            topologicalOrder = [];

            % Функция для DFS
            function visit(nodeIdx)
                if visited(nodeIdx)
                    return;
                end
                visited(nodeIdx) = true;

                % Рекурсивно посещаем все вершины, из которых есть ребра в текущую
                currentNode = obj.ListOfNodes(nodeIdx);
                incomingNeighbors = obj.getIncomingNeighbors(currentNode);

                for k = 1:numel(incomingNeighbors)
                    neighbor = incomingNeighbors(k);
                    neighborIdx = find(obj.ListOfNodes == neighbor, 1);
                    if ~isempty(neighborIdx)
                        visit(neighborIdx);
                    end
                end

                topologicalOrder = [nodeIdx, topologicalOrder];
            end

            % Обход всех вершин
            for i = 1:num_nodes
                if ~visited(i)
                    visit(i);
                end
            end
        end
    end

end


