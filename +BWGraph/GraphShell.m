classdef GraphShell < handle

    properties( Access = private)
        BetaGenerator          % Объект для генерации бета
    end

    properties
        ListOfNodes BWGraph.Node  % Вектор всех Node
        NumStages (1,1) double {mustBePositive, mustBeInteger} = 1  % K этапов Forward
    end

    properties (Access = private)
        fi_result
        numOfWhiteNodes
        numOfBlackNodes
        % Кеш прямого прохода: M_inv и ConstVec не меняются между
        % примерами в пределах одной эпохи (зависят только от α, β)
        cachedM_inv              % (D - A_in)⁻¹
        cachedConstVec           % B_in - B_out
        cachedEdgeHash   = ''    % хеш параметров рёбер для инвалидации
        % Кеш BPTT: промежуточные значения для обратного прохода по этапам
        bptt_F_stages             % cell{K}: F_vector каждого этапа
        bptt_ctx_stages           % cell{K}: ctx_i для каждого узла (0 если нет контекста)
        bptt_raw_stages           % cell{K}: raw_i (до активации) для каждого узла
        bptt_baseInputs           % cell{numNodes}: базовые входы (без контекста)
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
            % 3.1. Восстановление BetaGenerator
            betaGen = [];

            if isfield(modelData, 'Generators')
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

            % Собираем NodeWeight из загруженных Gamma
            loadedGammas = arrayfun(@(n) n.Gamma, nodes);

            % Создаем объект GraphShell
            loadedGraph = BWGraph.GraphShell(betaGen, loadedGammas, nodeCellArray{:});

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
            if isempty(loadedGraph.BetaGenerator)
                warning('GraphShell:MissingGenerators', ...
                    'Some generators were not loaded properly');
            end

            disp(['GraphShell successfully loaded from ' fullFilename]);
        end

    end

    methods (Access = public)
        function obj = GraphShell(BetaGenerator, NodeWeight, varargin)

            if ~isa(BetaGenerator, 'BWGraph.RandomGenerator.IRandomGen')
                error(['Генератор для beta параметров должен ' ...
                    'реализовывать интерфейс BWGraph.RandomGenerator.IRandomGen'])
            end

            obj.ListOfNodes = BWGraph.Node.empty(0, 1);
            obj.fi_result = [];
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

            % Генерируем beta, alpha — через GenerateTopologyAwareAlpha (вызывается извне)
            BetaGenerator.Generate(obj);
            % α инициализируются нулём — пользователь или Trainer должны вызвать GenerateTopologyAwareAlpha

            % Обновляем счётчики узлов
            obj.numOfBlackNodes = numel(obj.GetBlackNodesIndices);
            obj.numOfWhiteNodes = numel(obj.GetWhiteNodesIndices);

            % Устанавливаем гамма-параметры
            gammas = num2cell(NodeWeight);
            [obj.ListOfNodes.Gamma] = gammas{:};
        end

        function GenerateTopologyAwareAlpha(obj, SafetyFactor)
            % Публичная генерация α с учётом топологии.
            % Гарантирует Σα_in(v) ≤ SafetyFactor·(1+Σα_out(v)).
            % Вызывается из Trainer или вручную при использовании GraphShell без обучения.
            arguments
                obj          BWGraph.GraphShell
                SafetyFactor (1,1) double = 0.8
            end

            n = numel(obj.ListOfNodes);
            if n == 0, return; end

            incomingEdges = cell(n, 1);
            for i = 1:n
                incomingEdges{i} = obj.getIncomingEdges(obj.ListOfNodes(i));
            end

            for i = 1:n
                edges = obj.ListOfNodes(i).getOutEdges();
                for j = 1:numel(edges)
                    edges(j).Alfa = 0;
                end
            end

            for iter = 1:10
                changed = false;
                for v = 1:n
                    inEdges = incomingEdges{v};
                    nIn = numel(inEdges);
                    if nIn == 0, continue; end

                    outEdges = obj.ListOfNodes(v).getOutEdges();
                    D_v = 1 + sum([outEdges.Alfa]);
                    budget = SafetyFactor * D_v;

                    for j = 1:nIn
                        newAlpha = budget / nIn;
                        if abs(inEdges(j).Alfa - newAlpha) > 1e-8
                            inEdges(j).Alfa = newAlpha;
                            changed = true;
                        end
                    end
                end
                if ~changed, break; end
            end
        end

        % Прямой матричный подход с K этапами.
        % K=1 — классический плоский режим (без BPTT-оверхеда)
        function Forward(obj, Data)
            arguments
                obj     BWGraph.GraphShell
                Data    BWGraph.CustomMatrix.BWMatrix
            end

            numNodes = numel(obj.ListOfNodes);

            if Data.rowCount() ~= numNodes
                error('Размерность данных не соответствует числу вершин');
            end

            % Проверяем кеш: M_inv и ConstVec не меняются между этапами
            edgeHash = obj.computeEdgeHash();
            if isempty(obj.cachedEdgeHash) || ~strcmp(edgeHash, obj.cachedEdgeHash)
                [D, A_in, B_in, B_out, ~] = obj.buildSystemMatrices(Data);
                M = D - A_in;
                try
                    obj.cachedM_inv = M \ eye(numNodes);
                catch
                    obj.cachedM_inv = pinv(M);
                end
                obj.cachedConstVec = B_in - B_out;
                obj.cachedEdgeHash = edgeHash;
            end

            % --- Fast path: K=1 (классический режим, ноль оверхеда) ---
            if obj.NumStages <= 1
                L = zeros(numNodes, 1);
                for i = 1:numNodes
                    L(i) = obj.ListOfNodes(i).calcNodeFunc(Data.getRow(i));
                end
                F_vector = obj.cachedM_inv * (L + obj.cachedConstVec);
                for i = 1:numNodes
                    obj.ListOfNodes(i).setFResult(F_vector(i));
                end
                obj.fi_result = F_vector;
                return;
            end

            % --- Multi-stage path: K > 1 ---
            K = obj.NumStages;

            % Определяем, поддерживает ли каждая вершина контекст
            useContext = false(numNodes, 1);
            for i = 1:numNodes
                nf = obj.ListOfNodes(i).getNodeFunction();
                if ~isempty(nf)
                    useContext(i) = nf.SupportsContext();
                end
            end
            anyContext = any(useContext);

            % Предзагрузка базовых входных данных
            baseInputs = cell(numNodes, 1);
            for i = 1:numNodes
                baseInputs{i} = Data.getRow(i);
            end

            % Инициализация кеша BPTT
            if anyContext
                obj.bptt_F_stages = cell(K, 1);
                obj.bptt_ctx_stages = cell(K, 1);
                obj.bptt_raw_stages = cell(K, 1);
                obj.bptt_baseInputs = baseInputs;
            end

            % Многоэтапный проход
            F_prev = zeros(numNodes, 1);
            for stage = 1:K
                L = zeros(numNodes, 1);
                raw_store = zeros(numNodes, 1);
                ctx_store = cell(numNodes, 1);

                for i = 1:numNodes
                    augInput = baseInputs{i};
                    node = obj.ListOfNodes(i);

                    if stage > 1 && anyContext && useContext(i) && node.GammaCtx > 0
                        inEdges = obj.getIncomingEdges(node);
                        nIn = numel(inEdges);
                        if nIn > 0
                            ctx_vec = zeros(nIn, 1);
                            for e = 1:nIn
                                src = inEdges(e).SourceNode;
                                srcIdx = find(obj.ListOfNodes == src, 1);
                                if ~isempty(srcIdx)
                                    ctx_vec(e) = node.GammaCtx * F_prev(srcIdx);
                                end
                            end
                            ctx_store{i} = ctx_vec;
                            augInput = node.getNodeFunction().AugmentInput(baseInputs{i}, ctx_vec);
                        end
                    end

                    raw_store(i) = node.Gamma * node.calcRawCoreFunction(augInput);
                    L(i) = node.calcNodeFunc(augInput);
                end

                F_vector = obj.cachedM_inv * (L + obj.cachedConstVec);

                if anyContext
                    obj.bptt_F_stages{stage} = F_vector;
                    obj.bptt_ctx_stages{stage} = ctx_store;
                    obj.bptt_raw_stages{stage} = raw_store;
                end

                F_prev = F_vector;
            end

            % Сохранение финального F
            for i = 1:numNodes
                obj.ListOfNodes(i).setFResult(F_vector(i));
            end
            obj.fi_result = F_vector;
        end

        function invalidateForwardCache(obj)
            obj.cachedEdgeHash = '';
            obj.bptt_F_stages = {};
            obj.bptt_ctx_stages = {};
            obj.bptt_raw_stages = {};
        end

        function dGammaCtx = BackpropContext(obj, dF_final)
            % BPTT: обратное распространение градиента через этапы.
            % Вход:
            %   dF_final — dLoss/dF финального этапа (numNodes × 1)
            % Выход:
            %   dGammaCtx — dLoss/dGammaCtx от контекстного пути (numNodes × 1)

            numNodes = numel(obj.ListOfNodes);
            K = obj.NumStages;
            dGammaCtx = zeros(numNodes, 1);

            if K <= 1 || isempty(obj.bptt_F_stages)
                return;
            end

            dF = dF_final;

            for stage = K:-1:2
                dF_prev_add = zeros(numNodes, 1);

                for i = 1:numNodes
                    ctx_cell = obj.bptt_ctx_stages{stage};
                    if isempty(ctx_cell) || isempty(ctx_cell{i}), continue; end
                    ctx_vec = ctx_cell{i};

                    node = obj.ListOfNodes(i);
                    raw_i = obj.bptt_raw_stages{stage}(i);
                    actDeriv = node.getActivationDerivative(raw_i);

                    % ∂Core_i/∂ctx (вектор-строка 1×nIn)
                    nf = node.getNodeFunction();
                    dCore_dctx = nf.CalcContextDerivative(obj.bptt_baseInputs{i}, ctx_vec);

                    % ∂L_i/∂ctx = act'(raw) × γ_i × ∂Core/∂ctx (1×nIn)
                    dL_dctx = node.Gamma * actDeriv * dCore_dctx;

                    % ∂F_i/∂ctx = M_inv(i,i) × ∂L_i/∂ctx (1×nIn)
                    dF_dctx = obj.cachedM_inv(i, i) * dL_dctx;

                    % Градиент GammaCtx и пропагация на F_prev
                    inEdges = obj.getIncomingEdges(node);
                    nIn = numel(inEdges);
                    for e = 1:nIn
                        % d(ctx_j)/dGammaCtx = F_prev(src) = ctx_j / GammaCtx
                        dGammaCtx(i) = dGammaCtx(i) ...
                            + dF(i) * dF_dctx(e) * (ctx_vec(e) / node.GammaCtx);

                        % d(ctx_j)/dF_prev(src) = GammaCtx
                        src = inEdges(e).SourceNode;
                        srcIdx = find(obj.ListOfNodes == src, 1);
                        if ~isempty(srcIdx)
                            dF_prev_add(srcIdx) = dF_prev_add(srcIdx) ...
                                + dF(i) * dF_dctx(e) * node.GammaCtx;
                        end
                    end
                end

                dF = dF + dF_prev_add;
            end
        end

        function h = computeEdgeHash(obj)
            % Быстрый хеш всех α, β на рёбрах — для инвалидации кеша прямого прохода
            parts = {};
            for i = 1:numel(obj.ListOfNodes)
                edges = obj.ListOfNodes(i).getOutEdges();
                for j = 1:numel(edges)
                    parts{end+1} = sprintf('%.6g_%.6g', edges(j).Alfa, edges(j).Beta);
                end
            end
            h = strjoin(parts, '|');
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

            numNodes = numel(obj.ListOfNodes);

            % ВАЖНО: D должна быть матрицей n x n, а не вектором!
            D = zeros(numNodes, numNodes);  % Матрица
            A_in = zeros(numNodes, numNodes);
            B_in = zeros(numNodes, 1);
            B_out = zeros(numNodes, 1);
            L = zeros(numNodes, 1);

            for i = 1:numNodes
                currentNode = obj.ListOfNodes(i);

                % 1. Вычисляем L(i)
                if nargin > 1 && ~isempty(Data)
                    currentData = Data.getRow(i);
                    L(i) = currentNode.calcNodeFunc(currentData);
                else
                    L(i) = currentNode.getFResult();  % Или 0
                end

                % 2. Диагональный элемент D(i,i) = 1 + Σα_out (формула 2.6)
                outgoingEdges = currentNode.getOutEdges();
                if ~isempty(outgoingEdges)
                    D(i, i) = 1 + sum([outgoingEdges.Alfa]);
                else
                    D(i, i) = 1;
                end

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

            % Значение dL/dGamma
            dL_dGamma = v.computeLGammaDerivative(inputData);

            % Выходные ребра вершины
            outgoingEdges = v.getOutEdges();
            % 
            % denominator = 0;
            % for k = 1:numel(outgoingEdges)
            %     denominator = denominator + (outgoingEdges(k).Alfa + 1);
            % end

            if ~isempty(outgoingEdges)
                denominator = 1 + sum([outgoingEdges.Alfa]);
            else
                denominator = 1;
            end
            % 
            % denominator = sum(arrayfun(@(e) e.Alfa + 1, outgoingEdges));

            if abs(denominator) < eps
                error(['Знаменатель близок к нулю для вершины ', num2str(nodeIndex)]);
            end

            dF_dGamma = dL_dGamma / denominator;
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

            denominator = 1 + sum([outgoingEdges.Alfa]);

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

            denominator = 1 + sum([outgoingEdges.Alfa]);

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

            if ~isempty(outgoingEdges)
                denominator = 1 + sum([outgoingEdges.Alfa]);

                if abs(denominator) < eps
                    error(['Знаменатель близок к нулю для вершины ', num2str(nodeIndex)]);
                end

                % Формула (3.13)
                dF_dalpha = (F_u + alpha_e * dF_source_dalpha) / denominator;
            else
                dF_dalpha = F_u + alpha_e * dF_source_dalpha;
            end 
        end

        function dF_dbeta = computeIncomingBetaDerivativeDirect(obj, nodeIndex, edge, dF_source_dbeta)
            % Прямое вычисление ∂F_v/∂β для ВХОДЯЩЕГО ребра
            % Принимает уже вычисленную ∂F_u/∂β как параметр

            alpha_e = edge.Alfa;

            % Знаменатель для вершины v
            currentNode = obj.ListOfNodes(nodeIndex);
            outgoingEdges = currentNode.getOutEdges();
            
            if ~isempty(outgoingEdges)

                denominator = 1 + sum([outgoingEdges.Alfa]);

                if abs(denominator) < eps
                    error(['Знаменатель близок к нулю для вершины ', num2str(nodeIndex)]);
                end

                dF_dbeta = (1 + alpha_e * dF_source_dbeta) / denominator;

            else
                dF_dbeta = 1 + alpha_e * dF_source_dbeta;
            end
            
        end

        function DrawGraph_New(obj, titleStr, ax, hideEdgeLabels)
            % Метод для визуализации структуры графа с нелинейными параметрами
            % Если передан ax — рисует на заданных осях, иначе в новой фигуре

            % Определяем режим: полноэкранный или встроенный (subplot)
            embeddedMode = (nargin >= 3 && ~isempty(ax));
            if nargin < 4, hideEdgeLabels = false; end
            if embeddedMode
                axes(ax);
                cla(ax);
            else
                figure;
                set(gcf, 'Position', [100, 100, 1200, 800]);
            end

            % Создаем пустой ориентированный граф
            G = digraph();

            % Получаем все ID узлов
            nodeIDs = [obj.ListOfNodes.ID];
            nodeGammas = [obj.ListOfNodes.Gamma];
            nodeGammaCtxs = [obj.ListOfNodes.GammaCtx];

            % Добавляем узлы в граф (используем строковые ID)
            for i = 1:numel(nodeIDs)
                G = addnode(G, num2str(nodeIDs(i)));
            end

            % Добавляем рёбра с метками
            edgeLabelsShort = {};
            edgeLabelsFull = {};
            edgeSrcIDs = [];
            edgeTgtIDs = [];
            hasEdges = false;

            for i = 1:numel(obj.ListOfNodes)
                sourceNode = obj.ListOfNodes(i);
                edges = sourceNode.getOutEdges();

                for j = 1:numel(edges)
                    edge = edges(j);
                    targetNode = edge.TargetNode;

                    if any(nodeIDs == targetNode.ID)
                        G = addedge(G, num2str(sourceNode.ID), num2str(targetNode.ID));
                        edgeLabelsShort{end+1} = sprintf('α=%.3f β=%.3f', edge.Alfa, edge.Beta);
                        edgeLabelsFull{end+1} = sprintf('α=%.3f β=%.3f', edge.Alfa, edge.Beta);
                        edgeSrcIDs(end+1) = sourceNode.ID;
                        edgeTgtIDs(end+1) = targetNode.ID;
                        hasEdges = true;
                    else
                        warning('Target node ID %d not found in graph', targetNode.ID);
                    end
                end
            end

            % Метки узлов — только ID (чисто, без параметров)
            nodeLabels = arrayfun(@(x) sprintf(' v_{%d}', x), nodeIDs, 'UniformOutput', false);

            if embeddedMode
                nfs = 11; efs = 9; ms = 12; as = 10; lw = 1.5;
            else
                nfs = 18; efs = 11; ms = 20; as = 14; lw = 2.5;
            end

            if ~hasEdges
                h = plot(G, ...
                    'Layout', 'force', ...
                    'NodeLabel', nodeLabels, ...
                    'MarkerSize', ms, ...
                    'NodeFontSize', nfs, ...
                    'LineWidth', lw);
            else
                h = plot(G, ...
                    'Layout', 'layered', ...
                    'Direction', 'down', ...
                    'NodeLabel', nodeLabels, ...
                    'ArrowSize', as, ...
                    'LineWidth', lw, ...
                    'EdgeFontSize', efs, ...
                    'NodeFontSize', nfs, ...
                    'ArrowPosition', 0.9);

                h.EdgeColor = [0 0 0];
                h.EdgeAlpha = 0.8;
            end

            nodeColors = zeros(numel(obj.ListOfNodes), 3);
            for i = 1:numel(obj.ListOfNodes)
                if obj.ListOfNodes(i).getNodeType() == BWGraph.NodeColor.Black
                    nodeColors(i,:) = [0 0 0];
                else
                    nodeColors(i,:) = [0.9 0.9 0.9];
                end
            end

            h.NodeColor = [0 0 0];
            h.MarkerSize = ms;
            h.LineWidth = lw;

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
            if hasEdges && ~hideEdgeLabels
                h.EdgeLabel = edgeLabelsShort;
                set(h, 'EdgeLabelColor', [0.2 0.2 0.6]);

                if isprop(h, 'EdgeLabelRotation')
                    h.EdgeLabelRotation = 0;
                end
                if isprop(h, 'EdgeLabelBackgroundColor')
                    h.EdgeLabelBackgroundColor = [1 1 0.95];
                    h.EdgeLabelBackgroundAlpha = 0.8;
                end
            end

            % --- Таблица параметров вершин (γ, γ_C) ---
            nodeTableRows = cell(numel(obj.ListOfNodes) + 1, 1);
            if obj.NumStages > 1
                nodeTableRows{1} = sprintf('%-6s %-6s %-6s %s', 'Узел', 'Тип', '\gamma', '\gamma_C');
                sep = '---------------------------';
                nodeTableRows{2} = sep;
                for i = 1:numel(obj.ListOfNodes)
                    nt = obj.ListOfNodes(i).getNodeType();
                    typeStr = 'W'; if nt == BWGraph.NodeColor.Black, typeStr = 'B'; end
                    nodeTableRows{i+2} = sprintf('v_{%d}    %s     %5.2f  %5.2f', ...
                        nodeIDs(i), typeStr, nodeGammas(i), nodeGammaCtxs(i));
                end
            else
                nodeTableRows{1} = sprintf('%-6s %-6s %-6s', 'Узел', 'Тип', '\gamma');
                sep = '---------------------';
                nodeTableRows{2} = sep;
                for i = 1:numel(obj.ListOfNodes)
                    nt = obj.ListOfNodes(i).getNodeType();
                    typeStr = 'Б'; if nt == BWGraph.NodeColor.Black, typeStr = 'Ч'; end
                    nodeTableRows{i+2} = sprintf('v_{%d}    %s     %5.2f', ...
                        nodeIDs(i), typeStr, nodeGammas(i));
                end
            end
            nodeTableStr = strjoin(nodeTableRows, '\n');

            % --- Таблица параметров рёбер (α, β) ---
            if hasEdges
                edgeTableRows = cell(numel(edgeLabelsFull) + 1, 1);
                edgeTableRows{1} = sprintf('%-8s %-6s %-6s', 'Ребро', '\alpha', '\beta');
                edgeTableRows{2} = '--------------------';
                for e = 1:numel(edgeLabelsFull)
                    parts = strsplit(edgeLabelsFull{e}, ' ');
                    aVal = str2double(extractAfter(parts{1}, 'α='));
                    bVal = str2double(extractAfter(parts{2}, 'β='));
                    edgeTableRows{e+2} = sprintf('%d→%d     %5.2f  %5.2f', ...
                        edgeSrcIDs(e), edgeTgtIDs(e), aVal, bVal);
                end
                edgeTableStr = strjoin(edgeTableRows, '\n');
            end

            % --- Таблицы параметров (только в standalone-режиме) ---
            if ~embeddedMode
                xlims = xlim; ylims = ylim;
                tfs = 11;

                text(xlims(1) + 0.02*(xlims(2)-xlims(1)), ...
                     ylims(2) - 0.02*(ylims(2)-ylims(1)), ...
                     strrep(nodeTableStr, '_', '\_'), ...
                     'FontName', 'Courier New', 'FontSize', tfs, ...
                     'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
                     'BackgroundColor', [1 1 1], 'EdgeColor', [0.5 0.5 0.5], ...
                     'Margin', 3);

                if hasEdges
                    text(xlims(1) + 0.02*(xlims(2)-xlims(1)), ...
                         ylims(1) + 0.02*(ylims(2)-ylims(1)), ...
                         strrep(edgeTableStr, '_', '\_'), ...
                         'FontName', 'Courier New', 'FontSize', tfs, ...
                         'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', ...
                         'BackgroundColor', [1 1 1], 'EdgeColor', [0.5 0.5 0.5], ...
                         'Margin', 3);
                end
            end

            % Заголовок
            if nargin > 1 && ~isempty(titleStr)
                title(titleStr, 'FontSize', 12, 'FontWeight', 'bold');
            else
                title('Структура связей графа', 'FontSize', 12, 'FontWeight', 'bold');
            end

            nBlack = sum(arrayfun(@(nd) nd.getNodeType() == BWGraph.NodeColor.Black, obj.ListOfNodes));
            nWhite = numel(obj.ListOfNodes) - nBlack;

            % В полноэкранном режиме — информационные панели
            if ~embeddedMode
                infoText = sprintf('Узлов: %d (Черных: %d, Белых: %d)', ...
                    numel(obj.ListOfNodes), nBlack, nWhite);
                annotation('textbox', [0.02, 0.02, 0.3, 0.05], ...
                    'String', infoText, ...
                    'FontSize', 12, ...
                    'BackgroundColor', [0.95, 0.95, 0.95], ...
                    'EdgeColor', [0.5, 0.5, 0.5]);

                if hasEdges
                    paramText = {'Параметры рёбер:', ...
                        'α - линейный коэффициент', ...
                        'β - константное смещение'};
                    annotation('textbox', [0.7, 0.75, 0.2, 0.1], ...
                        'String', paramText, ...
                        'FontSize', 11, ...
                        'BackgroundColor', [0.95, 0.95, 0.95], ...
                        'EdgeColor', [0.5, 0.5, 0.5]);
                end
            end

            % Добавляем легенду для узлов
            if nBlack > 0 || nWhite > 0
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

                if nBlack > 0
                    legendHandles = [legendHandles, hBlack];
                    legendLabels = [legendLabels, 'Чёрные'];
                end
                if nWhite > 0
                    legendHandles = [legendHandles, hWhite];
                    legendLabels = [legendLabels, 'Белые'];
                end

                % Компактная легенда в правом нижнем углу
                lg = legend(legendHandles, legendLabels, ...
                    'Location', 'best', ...
                    'FontSize', 8, ...
                    'Box', 'off');

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

        % ===== Методы манипуляции топологией (структурный поиск) =====

        function addEdgeBetween(obj, sourceIdx, targetIdx, alpha, beta)
            % Добавляет направленное ребро от sourceIdx к targetIdx
            % Индексы — позиции в ListOfNodes (1-based)
            arguments
                obj         BWGraph.GraphShell
                sourceIdx   (1,1) double {mustBePositive, mustBeInteger}
                targetIdx   (1,1) double {mustBePositive, mustBeInteger}
                alpha       (1,1) double = 0
                beta        (1,1) double = 0
            end

            if sourceIdx == targetIdx
                error('Петли (sourceIdx == targetIdx) не допускаются');
            end
            if sourceIdx > numel(obj.ListOfNodes) || targetIdx > numel(obj.ListOfNodes)
                error('Индексы вершин выходят за пределы ListOfNodes');
            end
            if obj.hasEdge(sourceIdx, targetIdx)
                error('Ребро %d->%d уже существует', sourceIdx, targetIdx);
            end

            sourceNode = obj.ListOfNodes(sourceIdx);
            targetNode = obj.ListOfNodes(targetIdx);
            sourceNode.addEdge(targetNode);

            edge = sourceNode.getEdgeToTarget(targetNode);
            edge.Alfa = alpha;
            edge.Beta = beta;
        end

        function removeEdgeBetween(obj, sourceIdx, targetIdx)
            % Удаляет направленное ребро от sourceIdx к targetIdx
            arguments
                obj         BWGraph.GraphShell
                sourceIdx   (1,1) double {mustBePositive, mustBeInteger}
                targetIdx   (1,1) double {mustBePositive, mustBeInteger}
            end

            if sourceIdx > numel(obj.ListOfNodes) || targetIdx > numel(obj.ListOfNodes)
                error('Индексы вершин выходят за пределы ListOfNodes');
            end
            if ~obj.hasEdge(sourceIdx, targetIdx)
                error('Ребро %d->%d не существует', sourceIdx, targetIdx);
            end

            sourceNode = obj.ListOfNodes(sourceIdx);
            targetNode = obj.ListOfNodes(targetIdx);
            sourceNode.removeEdgeByTarget(targetNode);
        end

        function flag = hasEdge(obj, sourceIdx, targetIdx)
            % Проверяет существование ребра
            arguments
                obj         BWGraph.GraphShell
                sourceIdx   (1,1) double {mustBePositive, mustBeInteger}
                targetIdx   (1,1) double {mustBePositive, mustBeInteger}
            end

            if sourceIdx > numel(obj.ListOfNodes) || targetIdx > numel(obj.ListOfNodes)
                flag = false;
                return;
            end

            sourceNode = obj.ListOfNodes(sourceIdx);
            targetNode = obj.ListOfNodes(targetIdx);
            outMap = sourceNode.getOutEdgesMap();
            flag = isConfigured(outMap) && outMap.isKey(targetNode);
        end

        function pairs = getPossibleEdges(obj)
            % Возвращает N×2 матрицу пар (i,j), где ребра НЕТ и i≠j
            n = numel(obj.ListOfNodes);
            pairs = zeros(0, 2);
            for i = 1:n
                for j = 1:n
                    if i ~= j && ~obj.hasEdge(i, j)
                        pairs(end+1, :) = [i, j];
                    end
                end
            end
        end

        function pairs = getExistingEdges(obj)
            % Возвращает N×2 матрицу пар (i,j), где ребро ЕСТЬ
            n = numel(obj.ListOfNodes);
            pairs = zeros(0, 2);
            for i = 1:n
                sourceNode = obj.ListOfNodes(i);
                edges = sourceNode.getOutEdges();
                for k = 1:numel(edges)
                    targetNode = edges(k).TargetNode;
                    j = find(obj.ListOfNodes == targetNode, 1);
                    if ~isempty(j)
                        pairs(end+1, :) = [i, j];
                    end
                end
            end
        end

        function count = getTotalEdgeCount(obj)
            % Возвращает общее количество рёбер в графе
            count = 0;
            for i = 1:numel(obj.ListOfNodes)
                count = count + numel(obj.ListOfNodes(i).getOutEdges());
            end
        end

        function [alpha, beta] = generateEdgeParams(obj)
            % Генерирует пару (alpha, beta) теми же генераторами,
            % что использовались при начальной инициализации графа
            arguments
                obj BWGraph.GraphShell
            end

            import BWGraph.*;
            import coreFunctions.*;

            % Создаём временный граф из двух вершин с одним ребром
            tempSrc = Node(9999, 0, NodeColor.Black, [], "linear");
            tempDst = Node(9998, 0, NodeColor.Black, [], "linear");
            tempSrc.addEdge(tempDst);

            % Конструируем — beta заполняется генератором, alpha — топологически
            GraphShell(obj.BetaGenerator, ...
                [1, 1], tempSrc, tempDst);

            edges = tempSrc.getOutEdges();
            alpha = edges(1).Alfa;
            beta  = edges(1).Beta;
        end

        function [isStable, msg] = checkStability(obj)
            % Проверяет условия устойчивости (3.3) из рукописи для всех вершин
            % Возвращает:
            %   isStable — true если все вершины удовлетворяют условию
            %   msg     — описание первой найденной проблемы (пусто если стабилен)
            arguments
                obj BWGraph.GraphShell
            end

            isStable = true;
            msg = '';

            for i = 1:numel(obj.ListOfNodes)
                node = obj.ListOfNodes(i);

                % Знаменатель: 1 + Σα_out (базовая единица + сумма исходящих α)
                outEdges = node.getOutEdges();
                sumAlphaOut = 0;
                for k = 1:numel(outEdges)
                    sumAlphaOut = sumAlphaOut + outEdges(k).Alfa;
                end
                denominator = sumAlphaOut + 1; % 1 + Σα_out

                % Условие устойчивости (3.3): Σα_in < 1 + Σα_out
                % Базовая +1 гарантирует: для изолированной вершины 0 < 1 (стабильна),
                % для вершины без исходящих рёбер с одним входящим α=3: 3 < 1 (нестабильна)

                % Сумма входящих α
                sumAlphaIn = 0;
                for j = 1:numel(obj.ListOfNodes)
                    if j ~= i && obj.hasEdge(j, i)
                        [a, ~] = obj.getEdgeParams(j, i);
                        sumAlphaIn = sumAlphaIn + a;
                    end
                end

                if sumAlphaIn >= denominator
                    isStable = false;
                    msg = sprintf(['Вершина %d (позиция %d): нарушено условие устойчивости ' ...
                        '(Σα_in=%.4f ≥ 1+Σα_out=%.4f)'], ...
                        node.ID, i, sumAlphaIn, denominator);
                    return;
                end
            end
        end

        function [alpha, beta] = getEdgeParams(obj, sourceIdx, targetIdx)
            % Возвращает параметры (alpha, beta) конкретного ребра
            arguments
                obj         BWGraph.GraphShell
                sourceIdx   (1,1) double {mustBePositive, mustBeInteger}
                targetIdx   (1,1) double {mustBePositive, mustBeInteger}
            end

            if ~obj.hasEdge(sourceIdx, targetIdx)
                error('Ребро %d->%d не существует', sourceIdx, targetIdx);
            end

            sourceNode = obj.ListOfNodes(sourceIdx);
            targetNode = obj.ListOfNodes(targetIdx);
            edge = sourceNode.getEdgeToTarget(targetNode);
            alpha = edge.Alfa;
            beta = edge.Beta;
        end

        function cloned = clone(obj)
            % Создаёт глубокую копию графа (новые Node/Edge, те же генераторы)
            n = numel(obj.ListOfNodes);

            % 1. Создаём новые вершины с теми же свойствами
            newNodes = BWGraph.Node.empty(0, 1);
            for i = 1:n
                origNode = obj.ListOfNodes(i);
                nodeFunc = origNode.getNodeFunction();
                newNode = BWGraph.Node(...
                    origNode.ID, ...
                    origNode.getFResult(), ...
                    origNode.getNodeType(), ...
                    nodeFunc, ...
                    origNode.getActivationType());
                newNode.Gamma = origNode.Gamma;
                newNodes(end+1) = newNode;
            end

            % 2. Восстанавливаем рёбра
            for i = 1:n
                origNode = obj.ListOfNodes(i);
                edges = origNode.getOutEdges();
                for k = 1:numel(edges)
                    edge = edges(k);
                    targetNode = edge.TargetNode;
                    targetIdx = find(obj.ListOfNodes == targetNode, 1);
                    if ~isempty(targetIdx)
                        newNodes(i).addEdge(newNodes(targetIdx));
                        newEdge = newNodes(i).getEdgeToTarget(newNodes(targetIdx));
                        newEdge.Alfa = edge.Alfa;
                        newEdge.Beta = edge.Beta;
                    end
                end
            end

            % 3. Создаём GraphShell (конструктор вызовет Generate — параметры перезапишутся)
            gammas = [newNodes.Gamma];
            nodeCellArray = num2cell(newNodes);
            cloned = BWGraph.GraphShell(obj.BetaGenerator, gammas, nodeCellArray{:});

            % 3b. Восстанавливаем ПРАВИЛЬНЫЕ параметры рёбер (конструктор перегенерировал их)
            for i = 1:n
                origNode = obj.ListOfNodes(i);
                edges = origNode.getOutEdges();
                for k = 1:numel(edges)
                    edge = edges(k);
                    targetNode = edge.TargetNode;
                    targetIdx = find(obj.ListOfNodes == targetNode, 1);
                    if ~isempty(targetIdx)
                        clonedEdge = cloned.ListOfNodes(i).getEdgeToTarget(cloned.ListOfNodes(targetIdx));
                        clonedEdge.Alfa = edge.Alfa;
                        clonedEdge.Beta = edge.Beta;
                    end
                end
            end

            % 4. Копируем внутреннее состояние
            cloned.fi_result = obj.fi_result;
            cloned.cachedEdgeHash = ''; % Кеш прямого прохода должен перестроиться
            cloned.numOfWhiteNodes = obj.numOfWhiteNodes;
            cloned.numOfBlackNodes = obj.numOfBlackNodes;
        end

        function DrawNodeTable(obj, ax)
            % Таблица вершин: ID, тип, gamma, gCtx
            axes(ax); cla(ax); axis off; hold on;

            nodeIDs = [obj.ListOfNodes.ID];
            nodeGammas = [obj.ListOfNodes.Gamma];
            nodeGammaCtxs = [obj.ListOfNodes.GammaCtx];

            if obj.NumStages > 1
                nodeRows = {sprintf('%-6s %-5s %-10s %-10s', 'Узел', 'Тип', '\gamma', '\gamma_C');
                            repmat('-', 1, 27)};
                for i = 1:numel(obj.ListOfNodes)
                    nt = obj.ListOfNodes(i).getNodeType();
                    ts = 'Б'; if nt == BWGraph.NodeColor.Black, ts = 'Ч'; end
                    nodeRows{end+1} = sprintf('v_{%d}   %s     %5.2f  %5.2f', ...
                        nodeIDs(i), ts, nodeGammas(i), nodeGammaCtxs(i));
                end
            else
                nodeRows = {sprintf('%-6s %-5s %-10s', 'Узел', 'Тип', '\gamma');
                            repmat('-', 1, 20)};
                for i = 1:numel(obj.ListOfNodes)
                    nt = obj.ListOfNodes(i).getNodeType();
                    ts = 'Б'; if nt == BWGraph.NodeColor.Black, ts = 'Ч'; end
                    nodeRows{end+1} = sprintf('v_{%d}   %5s   %5.2f', ...
                        nodeIDs(i), ts, nodeGammas(i));
                end
            end
            nodeStr = strjoin(nodeRows, '\n');

            text(0.05, 0.95, strrep(nodeStr, '_', '_'), 'Units', 'normalized', ...
                'FontName', 'Courier New', 'FontSize', 11, ...
                'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
                'BackgroundColor', [1 1 1], 'EdgeColor', [0.5 0.5 0.5], 'Margin', 3, "Interpreter", "tex");
            title('Вершины', 'FontSize', 12);
        end

        function DrawEdgeTable(obj, ax)
            % Таблица рёбер: src->dst, alpha, beta
            axes(ax); cla(ax); axis off; hold on;

            edgeSrc = []; edgeTgt = []; edgeAl = []; edgeBt = [];
            for i = 1:numel(obj.ListOfNodes)
                edges = obj.ListOfNodes(i).getOutEdges();
                for j = 1:numel(edges)
                    e = edges(j);
                    edgeSrc(end+1) = e.SourceNode.ID;
                    edgeTgt(end+1) = e.TargetNode.ID;
                    edgeAl(end+1) = e.Alfa;
                    edgeBt(end+1) = e.Beta;
                end
            end

            if isempty(edgeSrc)
                text(0.5, 0.5, 'Нет рёбер', 'Units', 'normalized', ...
                    'HorizontalAlignment', 'center', 'FontSize', 10);
            else
                edgeRows = {sprintf('%-6s %-10s %-10s', 'Ребро', '\alpha', '\beta');
                            repmat('-', 1, 22)};
                for e = 1:numel(edgeSrc)
                    edgeRows{end+1} = sprintf('%d->%d  %5.2f  %5.2f', ...
                        edgeSrc(e), edgeTgt(e), edgeAl(e), edgeBt(e));
                end
                edgeStr = strjoin(edgeRows, '\n');

                text(0.05, 0.95, edgeStr, 'Units', 'normalized', ...
                    'FontName', 'Courier New', 'FontSize', 11, ...
                    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
                    'BackgroundColor', [1 1 1], 'EdgeColor', [0.5 0.5 0.5], 'Margin', 3);
            end
            title('Рёбра', 'FontSize', 12);
        end

        function DrawParamTables(obj, ax)
            % Таблица параметров вершин в нижнем левом углу графа
            axes(ax); hold on;

            nodeIDs = [obj.ListOfNodes.ID];
            nodeGammas = [obj.ListOfNodes.Gamma];
            nodeGammaCtxs = [obj.ListOfNodes.GammaCtx];

            if obj.NumStages > 1
                nodeRows = {sprintf('%-6s %-6s %-6s %s', 'Узел', 'Тип', '\gamma', '\gamma_C');
                            repmat('-', 1, 27)};
                for i = 1:numel(obj.ListOfNodes)
                    nt = obj.ListOfNodes(i).getNodeType();
                    ts = 'W'; if nt == BWGraph.NodeColor.Black, ts = 'B'; end
                    nodeRows{end+1} = sprintf('v_{%d}   %s      %5.2f  %5.2f', ...
                        nodeIDs(i), ts, nodeGammas(i), nodeGammaCtxs(i));
                end
            else
                nodeRows = {sprintf('%-6s %-6s %-6s', 'Узел', 'Тип', '\gamma');
                            repmat('-', 1, 21)};
                for i = 1:numel(obj.ListOfNodes)
                    nt = obj.ListOfNodes(i).getNodeType();
                    ts = 'W'; if nt == BWGraph.NodeColor.Black, ts = 'B'; end
                    nodeRows{end+1} = sprintf('v_{%d}   %s      %5.2f', ...
                        nodeIDs(i), ts, nodeGammas(i));
                end
            end
            nodeStr = strjoin(nodeRows, '\n');

            xlims = xlim; ylims = ylim;
            tfs = 10;
            text(xlims(1) + 0.02*(xlims(2)-xlims(1)), ...
                 ylims(1) + 0.02*(ylims(2)-ylims(1)), ...
                 strrep(nodeStr, '_', '\_'), ...
                 'FontName', 'Courier New', 'FontSize', tfs, ...
                 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', ...
                 'BackgroundColor', [1 1 1], 'EdgeColor', [0.5 0.5 0.5], 'Margin', 3);
        end

    end

    methods(Access = private)
        function topologicalOrder = getTopologicalOrder(obj)
            % Возвращает индексы вершин в топологическом порядке
            % Используется для правильного вычисления производных

            numNodes = numel(obj.ListOfNodes);
            visited = false(1, numNodes);
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

                topologicalOrder = [topologicalOrder, nodeIdx];
            end

            % Обход всех вершин
            for i = 1:numNodes
                if ~visited(i)
                    visit(i);
                end
            end
        end
    end

end


