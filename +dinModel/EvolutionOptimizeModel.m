classdef EvolutionOptimizeModel
    properties
        Model EvolutionModelShell
        ErrorArray
    end

    methods
        function obj = EvolutionOptimizeModel(Model)
            obj.Model = Model;
        end

        
        function Optimize(obj, XData, YData, BatchSize, LR, Beta1, Beta2, Eps)
            arguments
                obj EvolutionOptimizeModel
                XData (:,:,:,:)
                YData 
                BatchSize
                LR,
                Beta1,
                Beta2,
                Eps
            end
            
            % --- Инициализация параметров Adam ---
            learning_rate = LR;         % Шаг обучения (можно настроить)
            beta1 = Beta1;              % Для скользящего среднего градиента (момент)
            beta2 = Beta2;              % Для скользящего среднего квадратов градиента
            epsilon = Eps;              % Для численной стабильности
            batch_size = BatchSize;     % Размер мини-пакета
            
            num_vertex = obj.Model.getNumOfVertex();
            num_param = obj.Model.getNumOfParam();
            num_layers = obj.Model.getNumOfLayers();

            if size(XData,4) ~= size(YData,1)
                error("Количество входов и выходов должно совпадать!");
            end

            % --- Инициализация моментов Adam для al и bt ---
            % Предполагаем, что al и bt - матрицы размерности [numOfVertex, numOfVertex]
            m_al = zeros(num_vertex, num_vertex, num_layers);  % Первый момент (градиенты)
            v_al = zeros(num_vertex, num_vertex, num_layers);  % Второй момент (квадраты градиентов)
            m_bt = zeros(num_vertex, num_vertex, num_layers);
            v_bt = zeros(num_vertex, num_vertex, num_layers);
            m_tt = zeros(num_vertex, num_vertex, num_layers);
            v_tt = zeros(num_vertex, num_vertex, num_layers);

            % Инициализация массива ошибок
            obj.ErrorArray = zeros(size(XData,1), num_vertex, num_layers);
            % Количество образцов для пакетной настройки
            num_samples = size(XData,1);
            % Количество батчей
            num_batches = ceil(num_samples / batch_size);

            for batch_idx = 1:num_batches
                % Определяем индексы
                batch_start = (batch_idx-1)*batch_size + 1;
                batch_end = min(batch_idx*batch_size, num_samples);
                batch_indices = batch_start:batch_end;

                % Определяем матрицы для градиентов в пакете
                al_grad_batch = zeros(num_vertex, num_vertex, num_layers);
                bt_grad_batch = zeros(num_vertex, num_vertex, num_layers);
                tt_grad_batch = zeros(num_vertex, num_vertex, num_layers);

                for spl = 1:length(batch_indices)
                    % Извлекаем данные
                    xData = XData(:,:,:,spl);
                end

            end

        end

    end
end