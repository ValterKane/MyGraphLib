classdef GraphShellNonlinear < handle
    
    properties (Access = private)
        % Генераторы параметров
        AlphaGenerator
        BetaGenerator
        GammaGenerator
        DeltaGenerator
        
        % Кеши для производительности
        incomingEdgesCache
        outgoingEdgesCache
        incomingNeighborsCache
        
        % Вспомогательные структуры
        parameterCache
        derivativeCache
    end
    
    properties
        ListOfNodes        % Массив узлов
        fi_result          % Результаты прямого прохода
        numOfWhiteNodes    % Количество белых узлов
        numOfBlackNodes    % Количество черных узлов
    end
    
    methods
        function obj = GraphShellNonlinear(AlphaGenerator, BetaGenerator, GammaGenerator, DeltaGenerator, varargin)
            % Конструктор класса GraphShellNonlinear
            % Входные параметры:
            %   AlphaGenerator, BetaGenerator, GammaGenerator, DeltaGenerator - генераторы параметров
            %   varargin - список узлов графа
            
            % Проверка типов генераторов
            if ~isa(AlphaGenerator, 'BWGraph.RandomGenerator.IRandomGen') || ...
               ~isa(BetaGenerator, 'BWGraph.RandomGenerator.IRandomGen') || ...
               ~isa(GammaGenerator, 'BWGraph.RandomGenerator.IRandomGen') || ...
               ~isa(DeltaGenerator, 'BWGraph.RandomGenerator.IRandomGen')
                error('Все генераторы должны реализовывать интерфейс BWGraph.RandomGenerator.IRandomGen');
            end
            
            % Инициализация свойств
            obj.ListOfNodes = [];
            obj.fi_result = [];
            obj.AlphaGenerator = AlphaGenerator;
            obj.BetaGenerator = BetaGenerator;
            obj.GammaGenerator = GammaGenerator;
            obj.DeltaGenerator = DeltaGenerator;
            
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
            
            % Добавляем узлы
            obj.ListOfNodes = [obj.ListOfNodes; varargin{:}];
            
            % Генерируем значения параметров на рёбрах
            AlphaGenerator.Generate(obj);
            BetaGenerator.Generate(obj);
            GammaGenerator.Generate(obj);
            DeltaGenerator.Generate(obj);
            
            % Обновляем счётчики узлов
            obj.UpdateNodeCounters();
            
            % Инициализируем кеши
            obj.InitializeCaches();
        end
        
        function UpdateNodeCounters(obj)
            % Обновление счетчиков белых и черных узлов
            obj.numOfWhiteNodes = numel(obj.GetWhiteNodesIndices());
            obj.numOfBlackNodes = numel(obj.GetBlackNodesIndices());
        end
        
        function InitializeCaches(obj)
            % Инициализация кешей для производительности
            numNodes = numel(obj.ListOfNodes);
            
            obj.incomingEdgesCache = cell(numNodes, 1);
            obj.outgoingEdgesCache = cell(numNodes, 1);
            obj.incomingNeighborsCache = cell(numNodes, 1);
            
            for i = 1:numNodes
                node = obj.ListOfNodes(i);
                obj.incomingEdgesCache{i} = obj.getIncomingEdges(node);
                obj.outgoingEdgesCache{i} = node.getOutEdges();
                obj.incomingNeighborsCache{i} = obj.getIncomingNeighbors(node);
            end
            
            obj.parameterCache = containers.Map();
            obj.derivativeCache = containers.Map();
        end
        
        function [F_values, converged] = ForwardNonlinear(obj, Data, max_iterations, tolerance)
            % Прямой проход с нелинейными членами
            % Уравнение: F_v = (L_v + Σ[α_e*F_u + β_e + γ_e*F_u² + δ_e*F_v*F_u] - Σβ_e_out) / (Σ(α_e+1) - Σδ_e*F_u)
            
            arguments
                obj             BWGraph.NonLinearBWGraph.GraphShellNonlinear
                Data            BWGraph.CustomMatrix.BWMatrix
                max_iterations  (1,1) double = 100
                tolerance       (1,1) double = 1e-6
            end
            
            num_nodes = numel(obj.ListOfNodes);
            num_of_rows = Data.rowCount();
            
            if num_of_rows ~= num_nodes
                error('Входных значений должно быть столько же, сколько и вершин');
            end
            
            % Вычисляем L_v для всех узлов
            L_values = zeros(1, num_nodes);
            for i = 1:num_nodes
                currentData = Data.getRow(i);
                L_values(i) = obj.ListOfNodes(i).calcNodeFunc(currentData);
            end
            
            % Инициализация F_values
            F_values = L_values;
            
            % Итеративное уточнение
            converged = false;
            
            for iter = 1:max_iterations
                prev_values = F_values;
                
                for i = 1:num_nodes
                    currentNode = obj.ListOfNodes(i);
                    node_idx = i;
                    
                    % 1. Вычисляем Σ[α_e*F_u + β_e + γ_e*F_u² + δ_e*F_v*F_u] по входящим рёбрам
                    sum_incoming = 0;
                    incomingEdges = obj.incomingEdgesCache{node_idx};
                    
                    for k = 1:numel(incomingEdges)
                        edge = incomingEdges(k);
                        sourceNode = edge.SourceNode;
                        source_idx = find(obj.ListOfNodes == sourceNode, 1);
                        
                        if ~isempty(source_idx)
                            F_u = F_values(source_idx);
                            F_v = F_values(node_idx);
                            
                            % Используем все параметры
                            sum_incoming = sum_incoming + ...
                                edge.Alfa * F_u + ...
                                edge.Beta + ...
                                edge.Gamma * F_u^2 + ...
                                edge.Delta * F_v * F_u;
                        end
                    end
                    
                    % 2. Вычисляем Σβ_e по исходящим рёбрам
                    sum_beta_out = 0;
                    sum_alpha_plus_one = 0;
                    sum_delta_Fu = 0;
                    
                    outgoingEdges = obj.outgoingEdgesCache{node_idx};
                    
                    for k = 1:numel(outgoingEdges)
                        edge = outgoingEdges(k);
                        sum_beta_out = sum_beta_out + edge.Beta;
                        sum_alpha_plus_one = sum_alpha_plus_one + (edge.Alfa + 1);
                        
                        % Для знаменателя: Σδ_e*F_u
                        targetNode = edge.TargetNode;
                        target_idx = find(obj.ListOfNodes == targetNode, 1);
                        if ~isempty(target_idx)
                            F_u = F_values(target_idx);
                            sum_delta_Fu = sum_delta_Fu + edge.Delta * F_u;
                        end
                    end
                    
                    % Защита от деления на ноль
                    denominator = sum_alpha_plus_one - sum_delta_Fu;
                    if abs(denominator) < eps
                        error('Знаменатель в формуле F_v близок к нулю для вершины %d', currentNode.ID);
                    end
                    
                    % 3. Вычисляем новое значение F_v
                    F_values(node_idx) = (L_values(node_idx) + sum_incoming - sum_beta_out) / denominator;
                end
                
                % Проверка сходимости
                if max(abs(F_values - prev_values)) < tolerance
                    converged = true;
                    break;
                end
            end
            
            % Сохраняем результаты в узлы
            for i = 1:num_nodes
                obj.ListOfNodes(i).setFResult(F_values(i));
            end
            
            obj.fi_result = F_values;
        end
        
        function [dF_dalpha] = computeAlphaDerivativeRecursive(obj, node_idx, edge_idx, param_type, visited, cache_key)
            % Рекурсивный расчет частной производной по alpha
            % param_type: 'incoming' или 'outgoing'
            
            if nargin < 5
                visited = [];
                cache_key = sprintf('alpha_%d_%d_%s', node_idx, edge_idx, param_type);
            end
            
            % Проверка кеша
            if obj.derivativeCache.isKey(cache_key)
                dF_dalpha = obj.derivativeCache(cache_key);
                return;
            end
            
            % Проверка на цикл
            if ismember(node_idx, visited)
                dF_dalpha = 0;
                obj.derivativeCache(cache_key) = dF_dalpha;
                return;
            end
            
            visited = [visited, node_idx];
            currentNode = obj.ListOfNodes(node_idx);
            outgoingEdges = obj.outgoingEdgesCache{node_idx};
            incomingEdges = obj.incomingEdgesCache{node_idx};
            
            % Вычисляем S_out и Γ для текущей вершины
            S_out = 0;
            Gamma = 0;
            
            for k = 1:numel(outgoingEdges)
                edge = outgoingEdges(k);
                S_out = S_out + (edge.Alfa + 1);
                
                % Γ = Σδ_e*F_u
                targetNode = edge.TargetNode;
                target_idx = find(obj.ListOfNodes == targetNode, 1);
                if ~isempty(target_idx)
                    F_u = obj.ListOfNodes(target_idx).getFResult();
                    Gamma = Gamma + edge.Delta * F_u;
                end
            end
            
            denominator = S_out - Gamma;
            
            if strcmp(param_type, 'incoming')
                % Для входящих рёбер
                edge = incomingEdges(edge_idx);
                sourceNode = edge.SourceNode;
                source_idx = find(obj.ListOfNodes == sourceNode, 1);
                
                if isempty(source_idx)
                    dF_dalpha = 0;
                else
                    F_u = obj.ListOfNodes(source_idx).getFResult();
                    
                    % Рекурсивный расчет dF_u/dalpha
                    % Находим позицию этого ребра в исходящих рёбрах sourceNode
                    sourceOutgoing = obj.outgoingEdgesCache{source_idx};
                    source_edge_idx = find(arrayfun(@(e) e.TargetNode == currentNode, sourceOutgoing), 1);
                    
                    if ~isempty(source_edge_idx)
                        new_cache_key = sprintf('alpha_%d_%d_incoming', source_idx, source_edge_idx);
                        dF_u_dalpha = obj.computeAlphaDerivativeRecursive(source_idx, source_edge_idx, 'incoming', visited, new_cache_key);
                    else
                        dF_u_dalpha = 0;
                    end
                    
                    % Формула: dF_v/dalpha_e = (F_u + alpha_e * dF_u/dalpha_e + ...) / denominator
                    % Учитываем все влияния: α, γ, δ
                    gamma_term = 2 * edge.Gamma * F_u * dF_u_dalpha;
                    delta_term = edge.Delta * (obj.fi_result(node_idx) * dF_u_dalpha + ...
                        obj.ListOfNodes(node_idx).getFResult() * dF_u_dalpha);
                    
                    numerator = F_u + edge.Alfa * dF_u_dalpha + gamma_term + delta_term;
                    dF_dalpha = numerator / denominator;
                end
                
            else % 'outgoing'
                % Для исходящих рёбер
                edge = outgoingEdges(edge_idx);
                F_v = obj.ListOfNodes(node_idx).getFResult();
                
                % Влияние через знаменатель
                dS_out_dalpha = 1;  % производная S_out по alpha_e
                dGamma_dalpha = 0;  % gamma не зависит от alpha
                
                denominator_derivative = dS_out_dalpha - dGamma_dalpha;
                
                % Формула: dF_v/dalpha_e = -F_v * (dS_out/dalpha_e - dGamma/dalpha_e) / denominator^2
                dF_dalpha = -F_v * denominator_derivative / (denominator^2);
            end
            
            % Кешируем результат
            obj.derivativeCache(cache_key) = dF_dalpha;
        end
        
        function [dF_dbeta] = computeBetaDerivativeRecursive(obj, node_idx, edge_idx, param_type, visited, cache_key)
            % Рекурсивный расчет частной производной по beta
            
            if nargin < 5
                visited = [];
                cache_key = sprintf('beta_%d_%d_%s', node_idx, edge_idx, param_type);
            end
            
            if obj.derivativeCache.isKey(cache_key)
                dF_dbeta = obj.derivativeCache(cache_key);
                return;
            end
            
            if ismember(node_idx, visited)
                dF_dbeta = 0;
                obj.derivativeCache(cache_key) = dF_dbeta;
                return;
            end
            
            visited = [visited, node_idx];
            currentNode = obj.ListOfNodes(node_idx);
            outgoingEdges = obj.outgoingEdgesCache{node_idx};
            incomingEdges = obj.incomingEdgesCache{node_idx};
            
            % Вычисляем S_out и Γ
            S_out = 0;
            Gamma = 0;
            
            for k = 1:numel(outgoingEdges)
                edge = outgoingEdges(k);
                S_out = S_out + (edge.Alfa + 1);
                
                targetNode = edge.TargetNode;
                target_idx = find(obj.ListOfNodes == targetNode, 1);
                if ~isempty(target_idx)
                    F_u = obj.ListOfNodes(target_idx).getFResult();
                    Gamma = Gamma + edge.Delta * F_u;
                end
            end
            
            denominator = S_out - Gamma;
            
            if strcmp(param_type, 'incoming')
                % Для входящих рёбер
                edge = incomingEdges(edge_idx);
                sourceNode = edge.SourceNode;
                source_idx = find(obj.ListOfNodes == sourceNode, 1);
                
                if isempty(source_idx)
                    dF_dbeta = 0;
                else
                    % Рекурсивный расчет dF_u/dbeta
                    sourceOutgoing = obj.outgoingEdgesCache{source_idx};
                    source_edge_idx = find(arrayfun(@(e) e.TargetNode == currentNode, sourceOutgoing), 1);
                    
                    if ~isempty(source_edge_idx)
                        new_cache_key = sprintf('beta_%d_%d_incoming', source_idx, source_edge_idx);
                        dF_u_dbeta = obj.computeBetaDerivativeRecursive(source_idx, source_edge_idx, 'incoming', visited, new_cache_key);
                    else
                        dF_u_dbeta = 0;
                    end
                    
                    % Формула: dF_v/dbeta_e = (1 + alpha_e*dF_u/dbeta_e + ...) / denominator
                    gamma_term = 2 * edge.Gamma * obj.ListOfNodes(source_idx).getFResult() * dF_u_dbeta;
                    delta_term = edge.Delta * (obj.fi_result(node_idx) * dF_u_dbeta + ...
                        obj.ListOfNodes(node_idx).getFResult() * dF_u_dbeta);
                    
                    numerator = 1 + edge.Alfa * dF_u_dbeta + gamma_term + delta_term;
                    dF_dbeta = numerator / denominator;
                end
                
            else % 'outgoing'
                % Для исходящих рёбер
                edge = outgoingEdges(edge_idx);
                
                % Влияние через числитель: -1
                numerator_derivative = -1;
                
                % Формула: dF_v/dbeta_e = -1 / denominator
                dF_dbeta = numerator_derivative / denominator;
            end
            
            obj.derivativeCache(cache_key) = dF_dbeta;
        end
        
        function [dF_dgamma] = computeGammaDerivativeRecursive(obj, node_idx, edge_idx, visited, cache_key)
            % Рекурсивный расчет частной производной по gamma
            % gamma есть только на входящих рёбрах
            
            if nargin < 4
                visited = [];
                cache_key = sprintf('gamma_%d_%d', node_idx, edge_idx);
            end
            
            if obj.derivativeCache.isKey(cache_key)
                dF_dgamma = obj.derivativeCache(cache_key);
                return;
            end
            
            if ismember(node_idx, visited)
                dF_dgamma = 0;
                obj.derivativeCache(cache_key) = dF_dgamma;
                return;
            end
            
            visited = [visited, node_idx];
            currentNode = obj.ListOfNodes(node_idx);
            incomingEdges = obj.incomingEdgesCache{node_idx};
            outgoingEdges = obj.outgoingEdgesCache{node_idx};
            
            % Вычисляем S_out и Γ
            S_out = 0;
            Gamma = 0;
            
            for k = 1:numel(outgoingEdges)
                edge = outgoingEdges(k);
                S_out = S_out + (edge.Alfa + 1);
                
                targetNode = edge.TargetNode;
                target_idx = find(obj.ListOfNodes == targetNode, 1);
                if ~isempty(target_idx)
                    F_u = obj.ListOfNodes(target_idx).getFResult();
                    Gamma = Gamma + edge.Delta * F_u;
                end
            end
            
            denominator = S_out - Gamma;
            
            edge = incomingEdges(edge_idx);
            sourceNode = edge.SourceNode;
            source_idx = find(obj.ListOfNodes == sourceNode, 1);
            
            if isempty(source_idx)
                dF_dgamma = 0;
            else
                F_u = obj.ListOfNodes(source_idx).getFResult();
                
                % Рекурсивный расчет dF_u/dgamma
                sourceOutgoing = obj.outgoingEdgesCache{source_idx};
                source_edge_idx = find(arrayfun(@(e) e.TargetNode == currentNode, sourceOutgoing), 1);
                
                if ~isempty(source_edge_idx)
                    new_cache_key = sprintf('gamma_%d_%d', source_idx, source_edge_idx);
                    dF_u_dgamma = obj.computeGammaDerivativeRecursive(source_idx, source_edge_idx, visited, new_cache_key);
                else
                    dF_u_dgamma = 0;
                end
                
                % Формула: dF_v/dgamma_e = (F_u² + alpha_e*dF_u/dgamma_e + ...) / denominator
                alpha_term = edge.Alfa * dF_u_dgamma;
                gamma_term = edge.Gamma * 2 * F_u * dF_u_dgamma;
                delta_term = edge.Delta * (obj.fi_result(node_idx) * dF_u_dgamma + ...
                    obj.ListOfNodes(node_idx).getFResult() * dF_u_dgamma);
                
                numerator = F_u^2 + alpha_term + gamma_term + delta_term;
                dF_dgamma = numerator / denominator;
            end
            
            obj.derivativeCache(cache_key) = dF_dgamma;
        end
        
        function [dF_ddelta] = computeDeltaDerivativeRecursive(obj, node_idx, edge_idx, visited, cache_key)
            % Рекурсивный расчет частной производной по delta
            % delta есть только на входящих рёбрах
            
            if nargin < 4
                visited = [];
                cache_key = sprintf('delta_%d_%d', node_idx, edge_idx);
            end
            
            if obj.derivativeCache.isKey(cache_key)
                dF_ddelta = obj.derivativeCache(cache_key);
                return;
            end
            
            if ismember(node_idx, visited)
                dF_ddelta = 0;
                obj.derivativeCache(cache_key) = dF_ddelta;
                return;
            end
            
            visited = [visited, node_idx];
            currentNode = obj.ListOfNodes(node_idx);
            incomingEdges = obj.incomingEdgesCache{node_idx};
            outgoingEdges = obj.outgoingEdgesCache{node_idx};
            
            % Вычисляем S_out и Γ
            S_out = 0;
            Gamma = 0;
            
            for k = 1:numel(outgoingEdges)
                edge = outgoingEdges(k);
                S_out = S_out + (edge.Alfa + 1);
                
                targetNode = edge.TargetNode;
                target_idx = find(obj.ListOfNodes == targetNode, 1);
                if ~isempty(target_idx)
                    F_u = obj.ListOfNodes(target_idx).getFResult();
                    Gamma = Gamma + edge.Delta * F_u;
                end
            end
            
            denominator = S_out - Gamma;
            
            edge = incomingEdges(edge_idx);
            sourceNode = edge.SourceNode;
            source_idx = find(obj.ListOfNodes == sourceNode, 1);
            
            if isempty(source_idx)
                dF_ddelta = 0;
            else
                F_u = obj.ListOfNodes(source_idx).getFResult();
                F_v = obj.ListOfNodes(node_idx).getFResult();
                
                % Рекурсивный расчет dF_u/ddelta
                sourceOutgoing = obj.outgoingEdgesCache{source_idx};
                source_edge_idx = find(arrayfun(@(e) e.TargetNode == currentNode, sourceOutgoing), 1);
                
                if ~isempty(source_edge_idx)
                    new_cache_key = sprintf('delta_%d_%d', source_idx, source_edge_idx);
                    dF_u_ddelta = obj.computeDeltaDerivativeRecursive(source_idx, source_edge_idx, visited, new_cache_key);
                else
                    dF_u_ddelta = 0;
                end
                
                % Формула: dF_v/ddelta_e = (F_v*F_u + alpha_e*dF_u/ddelta_e + ...) / denominator
                % + дополнительный член от знаменателя
                
                alpha_term = edge.Alfa * dF_u_ddelta;
                gamma_term = edge.Gamma * 2 * F_u * dF_u_ddelta;
                delta_term = edge.Delta * (F_v * dF_u_ddelta + F_u * dF_u_ddelta);
                
                numerator = F_v * F_u + alpha_term + gamma_term + delta_term;
                
                % Влияние через знаменатель: -F_v * (-F_u) / denominator^2
                denominator_effect = F_v * F_u / (denominator^2);
                
                dF_ddelta = (numerator / denominator) + denominator_effect;
            end
            
            obj.derivativeCache(cache_key) = dF_ddelta;
        end
        
        function ClearDerivativeCache(obj)
            % Очистка кеша производных
            obj.derivativeCache = containers.Map();
        end
        
        function whiteNodeIndices = GetWhiteNodesIndices(obj)
            % Получение индексов белых узлов
            whiteNodeIndices = find(arrayfun(@(n) n.getNodeType() == BWGraph.NodeColor.White, obj.ListOfNodes));
        end
        
        function blackNodeIndices = GetBlackNodesIndices(obj)
            % Получение индексов черных узлов
            blackNodeIndices = find(arrayfun(@(n) n.getNodeType() == BWGraph.NodeColor.Black, obj.ListOfNodes));
        end
        
        function incomingEdges = getIncomingEdges(obj, targetNode)
            % Получение входящих рёбер для узла
            if ~isa(targetNode, 'BWGraph.Node')
                error('targetNode должен быть объектом класса Node.');
            end
            
            incomingEdges = BWGraph.Edge.empty(0, 1);
            
            for i = 1:numel(obj.ListOfNodes)
                currentNode = obj.ListOfNodes(i);
                outEdges = currentNode.getOutEdges();
                
                for j = 1:numel(outEdges)
                    edge = outEdges(j);
                    if edge.TargetNode == targetNode
                        incomingEdges(end+1) = edge;
                    end
                end
            end
        end

        function DrawGraph(obj, titleStr)
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

                        % Формируем метку с четырьмя параметрами
                        % Проверяем наличие параметров γ и δ
                        if isprop(edge, 'Gamma') && isprop(edge, 'Delta')
                            edgeLabels{end+1} = sprintf('α=%.2f\n β=%.2f\n γ=%.2f\n δ=%.2f', ...
                                edge.Alfa, edge.Beta, edge.Gamma, edge.Delta);
                        else
                            % Если параметры γ и δ не определены, показываем только α и β
                            edgeLabels{end+1} = sprintf('α=%.2f\n β=%.2f', edge.Alfa, edge.Beta);
                        end
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
                    'β - константное смещение', ...
                    'γ - квадратичный коэффициент', ...
                    'δ - кросс-коэффициент'};

                annotation('textbox', [0.7, 0.75, 0.25, 0.2], ...
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
        
        function incomingNodes = getIncomingNeighbors(obj, targetNode)
            % Получение соседей по входящим рёбрам
            if ~isa(targetNode, 'BWGraph.Node')
                error('targetNode должен быть объектом класса Node.');
            end
            
            incomingNodes = BWGraph.Node.empty(0, 1);
            
            for i = 1:numel(obj.ListOfNodes)
                node = obj.ListOfNodes(i);
                oo = node.getOutEdgesMap();
                if isKey(oo, targetNode)
                    incomingNodes(end+1) = node;
                end
            end
        end
        
        function results = GetModelResults(obj)
            % Получение результатов модели
            results = arrayfun(@(n) n.getFResult(), obj.ListOfNodes);
        end
        
        function results = GetCurrentResult(obj, XData)
            % Получение текущих результатов для данных
            obj.ForwardNonlinear(XData);
            results = obj.GetModelResults();
        end
    end
end