% ===== Гибридный генератор Beta =====
classdef HybridBetaGenerator < BWGraph.RandomGenerator.IRandomGen
      
    properties (Access = private)
        ScaleFactor    % Общий масштаб для β
        L_awareMode    % Режим учета L_v (true/false)
        BiasType       % Тип смещения: 'zero', 'small', 'adaptive'
    end
    
    methods
        function obj = HybridBetaGenerator(scale_factor, l_aware_mode, bias_type)
            % Конструктор с параметрами:
            % scale_factor = 0.01..0.1 (0.05 по умолчанию)
            % l_aware_mode = true/false (false по умолчанию)
            % bias_type = 'zero', 'small', 'adaptive' ('small' по умолчанию)
            arguments
                scale_factor (1,1) double = 0.05
                l_aware_mode (1,1) logical = false
                bias_type (1,1) string = "small"
            end
            
            obj.ScaleFactor = max(0.001, min(0.2, scale_factor));
            obj.L_awareMode = l_aware_mode;
            
            % Проверяем допустимые значения bias_type
            valid_types = ["zero", "small", "adaptive"];
            if ~ismember(bias_type, valid_types)
                error('Invalid bias_type. Use: "zero", "small", or "adaptive"');
            end
            obj.BiasType = bias_type;
        end
        
        function Generate(obj, graph_shell)
            arguments
                obj 
                graph_shell BWGraph.GraphShell
            end 
            
            listOfNodes = graph_shell.ListOfNodes;
            n_nodes = numel(listOfNodes);
            
            % 1. Собираем информацию о вершинах для adaptive режима
            node_info = struct();
            for i = 1:n_nodes
                currentNode = listOfNodes(i);
                node_info(i).is_white = graph_shell.IsWhiteVertice(i);
                node_info(i).L_function = currentNode.getNodeFunction();
                
                % Если режим L-aware и вершина белая, оцениваем L_v
                if obj.L_awareMode && node_info(i).is_white && ~isempty(node_info(i).L_function)
                    try
                        % Простая оценка масштаба L_v
                        sample_input = [10000,1006];
                        outputs = arrayfun(@(x) node_info(i).L_function.evaluate(x), sample_input);
                        node_info(i).L_std = std(outputs);
                        node_info(i).L_mean = mean(outputs);
                    catch
                        node_info(i).L_std = 1.0;
                        node_info(i).L_mean = 0.0;
                    end
                else
                    node_info(i).L_std = 1.0;
                    node_info(i).L_mean = 0.0;
                end
            end
            
            % 2. Вычисляем степени вершин
            out_degrees = zeros(1, n_nodes);
            for i = 1:n_nodes
                currentNode = listOfNodes(i);
                out_degrees(i) = numel(currentNode.getOutEdges());
            end
            
            % 3. Генерируем β для всех рёбер
            for i = 1:n_nodes
                currentNode = listOfNodes(i);
                currentEdges = currentNode.getOutEdges();
                
                for j = 1:numel(currentEdges)
                    edge = currentEdges(j);
                    
                    % Определяем базовое значение β в зависимости от типа
                    switch obj.BiasType
                        case "zero"
                            base_beta = 0.0;
                            
                        case "small"
                            % Малое случайное значение
                            base_beta = obj.ScaleFactor * (2*rand() - 1);
                            
                        case "adaptive"
                            % Адаптивное значение на основе свойств вершин
                            source_info = node_info(i);
                            
                            % Ищем целевую вершину
                            target_node = graph_shell.getTargetNode(edge);
                            target_idx = find(arrayfun(@(n) n == target_node, listOfNodes), 1);
                            
                            if ~isempty(target_idx)
                                target_info = node_info(target_idx);
                                
                                % Учитываем разницу между вершинами
                                if source_info.is_white && target_info.is_white
                                    % Белая → Белая: компенсация разницы средних
                                    beta_val = 0.01 * (target_info.L_mean - source_info.L_mean);
                                elseif source_info.is_white && ~target_info.is_white
                                    % Белая → Чёрная: небольшое отрицательное смещение
                                    beta_val = -0.005 * source_info.L_std;
                                elseif ~source_info.is_white && target_info.is_white
                                    % Чёрная → Белая: компенсация
                                    beta_val = 0.005 * target_info.L_std;
                                else
                                    % Чёрная → Чёрная: близко к нулю
                                    beta_val = 0.001 * (2*rand() - 1);
                                end
                                
                                base_beta = beta_val;
                            else
                                base_beta = obj.ScaleFactor * (2*rand() - 1);
                            end
                    end
                    
                    % Корректируем на основе степени исхода
                    if out_degrees(i) > 0
                        base_beta = base_beta / sqrt(out_degrees(i));
                    end
                    
                    % Ограничиваем диапазон
                    beta_val = max(-0.1, min(0.1, base_beta));
                    
                    edge.Beta = beta_val;
                end
            end
            
            % 4. Специальная обработка для вершин без исходящих рёбер
            for i = 1:n_nodes
                if out_degrees(i) == 0
                    currentNode = listOfNodes(i);
                    
                    % Для стоков устанавливаем β на входящих рёбрах
                    in_edges = graph_shell.getIncomingEdges(currentNode);
                    
                    for j = 1:numel(in_edges)
                        edge = in_edges(j);
                        
                        if isempty(edge.Beta)
                            % Устанавливаем небольшое положительное β для стоков
                            edge.Beta = 0.01 * rand();
                        end
                    end
                end
            end
            
            % 5. Балансировка: сумма β для исходящих рёбер вершины ≈ 0
            for i = 1:n_nodes
                currentNode = listOfNodes(i);
                out_edges = currentNode.getOutEdges();
                
                if numel(out_edges) > 1
                    betas = arrayfun(@(e) e.Beta, out_edges);
                    beta_sum = sum(betas);
                    
                    if abs(beta_sum) > 0.05
                        % Корректируем чтобы сумма была ближе к 0
                        correction = -beta_sum / numel(out_edges);
                        
                        for j = 1:numel(out_edges)
                            out_edges(j).Beta = out_edges(j).Beta + correction;
                            
                            % Ограничиваем
                            out_edges(j).Beta = max(-0.1, min(0.1, out_edges(j).Beta));
                        end
                    end
                end
            end
        end
    
        function data = save(obj)
            % Сериализация объекта в структуру
            data = struct();
            data.ClassName = class(obj);
            data.ScaleFactor = obj.ScaleFactor;
            data.L_awareMode = obj.L_awareMode;
            data.BiasType = obj.BiasType;
        end
        
        function load(obj, data)
            % Десериализация объекта из структуры
            if ~strcmp(data.ClassName, class(obj))
                error('Class mismatch during loading');
            end
            obj.ScaleFactor = data.ScaleFactor;
            obj.L_awareMode = data.L_awareMode;
            obj.BiasType = data.BiasType;
        end
    end

    methods (Static)
        function obj = createFromData(data)
            % Фабричный метод для создания экземпляра при загрузке
            obj = HybridBetaGenerator();
            obj.load(data);
        end
        
        function obj = createDefault()
            % Создание генератора с параметрами по умолчанию
            obj = HybridBetaGenerator(0.05, false, "small");
        end
        
        function obj = createForWhiteNodes()
            % Создание генератора, оптимизированного для белых вершин
            obj = HybridBetaGenerator(0.03, true, "adaptive");
        end
        
        function obj = createForStableInit()
            % Создание генератора для стабильной инициализации
            obj = HybridBetaGenerator(0.02, false, "zero");
        end
    end
    
    methods (Access = private)
        function target_node = getTargetNode(~, edge)
            % Вспомогательный метод для получения целевой вершины ребра
            % Нужно адаптировать под конкретную реализацию GraphShell
            target_node = edge.TargetNode; % Предполагаемая структура
        end
    end
end