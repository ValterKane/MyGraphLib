% ===== Гибридный генератор Alpha =====
classdef HybridAlphaGenerator < BWGraph.RandomGenerator.IRandomGen
      
    properties (Access = private)
        SafetyFactor      % Коэффициент запаса устойчивости
        XavierScale       % Масштаб для Xavier-подобной инициализации
        NoiseLevel        % Уровень случайного шума
    end
    
    methods
        function obj = HybridAlphaGenerator(safety_factor, xavier_scale, noise_level)
            % Конструктор с параметрами:
            % safety_factor = 0.7..0.9 (0.8 по умолчанию)
            % xavier_scale = 1.0..2.0 (1.4 по умолчанию)
            % noise_level = 0.0..0.1 (0.05 по умолчанию)
            arguments
                safety_factor (1,1) double = 0.8
                xavier_scale (1,1) double = 1.4
                noise_level (1,1) double = 0.05
            end
            
            obj.SafetyFactor = max(0.1, min(0.95, safety_factor));
            obj.XavierScale = max(0.5, min(3.0, xavier_scale));
            obj.NoiseLevel = max(0.0, min(0.2, noise_level));
        end
        
        function Generate(obj, graph_shell)
            arguments
                obj 
                graph_shell BWGraph.GraphShell
            end 
            
            listOfNodes = graph_shell.ListOfNodes;
            n_nodes = numel(listOfNodes);
            
            % 1. Вычисляем степени вершин
            out_degrees = zeros(1, n_nodes);
            in_degrees = zeros(1, n_nodes);
            
            for i = 1:n_nodes
                currentNode = listOfNodes(i);
                out_degrees(i) = numel(currentNode.getOutEdges());
                in_degrees(i) = numel(graph_shell.getIncomingNeighbors(currentNode));
            end
            
            % 2. Инициализируем α для всех рёбер
            alpha_map = containers.Map('KeyType', 'char', 'ValueType', 'double');
            
            % Проходим сначала по исходящим рёбрам
            for i = 1:n_nodes
                currentNode = listOfNodes(i);
                currentEdges = currentNode.getOutEdges();
                
                if ~isempty(currentEdges)
                    % Базовая Xavier-like инициализация для исходящих
                    fan_out = out_degrees(i);
                    incomNodes = graph_shell.getIncomingNeighbors(currentNode);
                    if ~isempty(incomNodes)
                        incomNodes = incomNodes.ID;
                        fan_in_mean = mean(in_degrees(incomNodes));
                        if isnan(fan_in_mean), fan_in_mean = 1; end

                        limit = obj.XavierScale * sqrt(6.0 / (fan_out + fan_in_mean + 1e-6));
                        base_alpha = limit;

                        % Распределяем по исходящим рёбрам
                        for j = 1:numel(currentEdges)
                            edge = currentEdges(j);
                            edge_key = sprintf('edge_%d_%d', i, j);

                            % Базовая инициализация + небольшой шум
                            alpha_val = base_alpha / fan_out;
                            alpha_val = alpha_val * (1 + obj.NoiseLevel * (2*rand() - 1));
                            alpha_val = max(0.001, min(0.95, alpha_val)); % Ограничиваем диапазон

                            alpha_map(edge_key) = alpha_val;
                            edge.Alfa = alpha_val;
                        end
                    end
                end
            end
            
            % 3. Инициализируем α для входящих рёбер (если ещё не инициализированы)
            for i = 1:n_nodes
                currentNode = listOfNodes(i);
                incoming_edges = graph_shell.getIncomingEdges(currentNode);
                
                for j = 1:numel(incoming_edges)
                    edge = incoming_edges(j);
                    
                    % Проверяем, инициализировано ли уже ребро
                    edge_found = false;
                    for k = 1:n_nodes
                        source_node = listOfNodes(k);
                        source_edges = source_node.getOutEdges();
                        for m = 1:numel(source_edges)
                            if source_edges(m) == edge
                                edge_found = true;
                                break;
                            end
                        end
                        if edge_found, break; end
                    end
                    
                    if ~edge_found
                        % Для входящих рёбер от вершин без исходящих
                        edge_key = sprintf('in_edge_%d_%d', i, j);
                        
                        % Делаем α для входящих меньше с запасом
                        source_idx = find(arrayfun(@(n) ismember(edge, n.getOutEdges()), listOfNodes));
                        if ~isempty(source_idx)
                            source_out_deg = out_degrees(source_idx);
                            target_in_deg = in_degrees(i);
                            
                            % Гарантируем условие устойчивости
                            safe_alpha = obj.SafetyFactor * 0.5 / max(target_in_deg, 1);
                            safe_alpha = min(safe_alpha, 0.8 / max(source_out_deg, 1));
                            
                            alpha_map(edge_key) = safe_alpha;
                            edge.Alfa = safe_alpha;
                        end
                    end
                end
            end
            
            % 4. Коррекция для гарантии устойчивости
            for i = 1:n_nodes
                currentNode = listOfNodes(i);
                
                % Суммируем α для входящих и исходящих
                sum_alpha_in = 0;
                sum_alpha_out = 0;
                
                % Входящие
                in_edges = graph_shell.getIncomingEdges(currentNode);
                for j = 1:numel(in_edges)
                    if ~isempty(in_edges(j).Alfa)
                        sum_alpha_in = sum_alpha_in + in_edges(j).Alfa;
                    end
                end
                
                % Исходящие
                out_edges = currentNode.getOutEdges();
                for j = 1:numel(out_edges)
                    if ~isempty(out_edges(j).Alfa)
                        sum_alpha_out = sum_alpha_out + out_edges(j).Alfa;
                    end
                end
                
                % Проверяем условие устойчивости
                if sum_alpha_in >= sum_alpha_out + 1 - 1e-6
                    % Масштабируем входящие α
                    scale_factor = obj.SafetyFactor * (sum_alpha_out + 1) / (sum_alpha_in + 1e-6);
                    
                    for j = 1:numel(in_edges)
                        if ~isempty(in_edges(j).Alfa)
                            in_edges(j).Alfa = in_edges(j).Alfa * scale_factor;
                        end
                    end
                end
            end
            
            % 5. Финальная нормализация
            for i = 1:n_nodes
                currentNode = listOfNodes(i);
                
                % Нормализуем исходящие α, чтобы сумма была разумной
                out_edges = currentNode.getOutEdges();
                if ~isempty(out_edges)
                    alphas = arrayfun(@(e) e.Alfa, out_edges);
                    alpha_sum = sum(alphas);
                    
                    if alpha_sum > 1.5
                        scale = 1.5 / alpha_sum;
                        for j = 1:numel(out_edges)
                            out_edges(j).Alfa = out_edges(j).Alfa * scale;
                        end
                    end
                end
            end
        end
    
        function data = save(obj)
            % Сериализация объекта в структуру
            data = struct();
            data.ClassName = class(obj);
            data.SafetyFactor = obj.SafetyFactor;
            data.XavierScale = obj.XavierScale;
            data.NoiseLevel = obj.NoiseLevel;
        end
        
        function load(obj, data)
            % Десериализация объекта из структуры
            if ~strcmp(data.ClassName, class(obj))
                error('Class mismatch during loading');
            end
            obj.SafetyFactor = data.SafetyFactor;
            obj.XavierScale = data.XavierScale;
            obj.NoiseLevel = data.NoiseLevel;
        end
    end

    methods (Static)
        function obj = createFromData(data)
            % Фабричный метод для создания экземпляра при загрузке
            obj = HybridAlphaGenerator();
            obj.load(data);
        end
        
        function obj = createDefault()
            % Создание генератора с параметрами по умолчанию
            obj = HybridAlphaGenerator(0.8, 1.4, 0.05);
        end
    end
end