classdef OptimizationMachine < handle

    properties (Access = private)
        % -- Массивы моментов для Adam --
        m_al
        v_al 
        m_bt 
        v_bt
        t
        % ------------------------------- 
        epochs_no_improve = 0;      % Фактическое количество эпох без улучшения
        best_al                     % Лучшие альфа-значения по результатам настройки
        best_bt                     % Лучшие бета-значения по результатам настройки
    end
    properties
        modelShell statModel.ModelShell       % Стационарная графовая модель
        errorArray                  % Массив ошибок обучения
        best_test_error = Inf;      % Поле для отслеживания улучшения ошибки
        patience = 25;              % Количество эпох без улучшения для ранней остановки
        min_delta = 0.001;          % Минимальное улучшение для сохранения модели
        learning_rate = 0.001;      % Начальный шаг обучения
        min_lr = 1e-6;              % Минимальный шаг обучения 
        lr_reduction_factor = 0.5;  % Степень редуцирования шага обучения
        num_epoch                   % Количество эпох обучения

        % --- Сохранение истории ошибок ---
        train_errors = [];
        test_errors = [];
        % ---------------------------------

        % -- Параметры настройки на плато ---
        % Параметры для случайного смещения
        plateau_count = 0;          % Счетчик плато
        max_plateau_count = 10;     % Максимальное количество плато перед остановкой
        random_shift_scale = 0.01;   % Масштаб случайного смещения параметров
        % -----------------------------------
    end

    methods (Access = public)
        function obj =  OptimizationMachine(modelShell)
            obj.modelShell = modelShell;
        end
   
        function model = GetModelShell(obj)
            model = obj.modelShell;
        end

        function Train(obj, XData_train, YData_train, XData_test, YData_test, ...
                BatchSize, LearningRate, Beta1, Beta2, Eps, VertexWeight, VertexImportance, Epoches)
            
            if size(XData_train,3) ~= size(YData_train,3)
                error("Количество входов и выходов в обучающей выборке должно совпадать!");
            elseif size(XData_train,1) ~= obj.modelShell.GetNumOfParameters() &&...
                    size(XData_train,2) ~= obj.modelShell.GetNumOfVertex()
                error("Для указанной ядровой функции входы обучающей выборки " + ...
                    "должны представлять собой матрицу (%d:%d)", ...
                    obj.modelShell.GetNumOfParameters(), obj.modelShell.GetNumOfVertex());
            elseif size(XData_test,3) ~= size(YData_test,3)
                error("Количество входов и выходов в тестовой выборке должно совпадать!");
            elseif size(XData_test,1) ~= obj.modelShell.GetNumOfParameters() &&...
                    size(XData_test,2) ~= obj.modelShell.GetNumOfVertex()
                error("Для указанной ядровой функции входы тестовой выборки" + ...
                    " должны представлять собой матрицу (%d:%d)", ...
                    obj.modelShell.GetNumOfParameters(), obj.modelShell.GetNumOfVertex());
            end
            
            % -- Применение настроек модели
            obj.num_epoch = Epoches;
            obj.learning_rate = LearningRate;

            % -- Основной цикл настройки
            for epoch = 1:obj.num_epoch
                % Динамическое уменьшение LR на основе плато
                if mod(epoch, 10) == 0
                    obj.learning_rate = max(obj.min_lr, obj.learning_rate * 0.98);
                end

                % Обучение на всей обучающей выборке
                obj.Optimize(XData_train, YData_train, BatchSize, ...
                    Beta1, Beta2, Eps, VertexWeight, VertexImportance);

                % Расчет ошибки на всей обучающей выборке
                train_error = 0;
                for i = 1:size(XData_train, 3)
                    obj.modelShell.Forward(XData_train(:, :, i));
                    model_value = obj.modelShell.GetResult();
                    % Метрика MAE
                    train_error = train_error + mean(abs(model_value - YData_train(1, :, i)));
                end
                train_error = train_error / size(XData_train, 3);

                obj.train_errors(end+1) = train_error;

                % Расчет ошибки на тестовой выборке
                test_error = 0;
                for i = 1:size(XData_test, 3)
                    obj.modelShell.Forward(XData_test(:, :, i));
                    model_value = obj.modelShell.GetResult();
                    % Метрика MAE
                    test_error = test_error + mean(abs(model_value - YData_test(1, :, i)));
                end
                test_error = test_error / size(XData_test, 3);
                obj.test_errors(end+1) = test_error;

                % Проверка улучшения на тестовой выборке
                if test_error < obj.best_test_error - obj.min_delta
                    obj.best_test_error = test_error;
                    obj.epochs_no_improve = 0;
                    % Сохраняем лучшие параметры
                    obj.best_al = obj.modelShell.GetAlfas();
                    obj.best_bt = obj.modelShell.GetBetas();
                else
                    obj.epochs_no_improve = obj.epochs_no_improve + 1;
                end

                % Уменьшение LR при застое
                if obj.epochs_no_improve > 0 && mod(obj.epochs_no_improve, 5) == 0
                    obj.learning_rate = max(obj.min_lr, obj.learning_rate * obj.lr_reduction_factor);
                    fprintf('Уменьшение LR до %.2e на эпохе %d\n', obj.learning_rate, epoch);
                end

                % Проверка на плато и случайное смещение параметров
                if obj.epochs_no_improve >= obj.patience
                    obj.plateau_count = obj.plateau_count + 1;
                    fprintf('Обнаружено плато ошибки (попытка %d из %d) на эпохе %d\n', obj.plateau_count, obj.max_plateau_count, epoch);

                    if obj.plateau_count >= obj.max_plateau_count
                        fprintf('Достигнуто максимальное количество плато (%d). Остановка обучения.\n', obj.max_plateau_count);
                        break;
                    else
                        % Случайное смещение параметров
                        current_al = obj.modelShell.GetAlfas();
                        current_bt = obj.modelShell.GetBetas();

                        % Генерируем случайные смещения
                        random_shift_al = (rand(size(current_al)) * 2 - 1) * obj.random_shift_scale;
                        random_shift_bt = (rand(size(current_bt)) * 2 - 1) * obj.random_shift_scale;

                        % Применяем смещения
                        new_al = current_al + random_shift_al;
                        new_bt = current_bt + random_shift_bt;

                        % Устанавливаем новые параметры
                        obj.modelShell.SetAlfaBeta(new_al, new_bt);

                        % Сброс счетчиков и learning rate
                        obj.epochs_no_improve = 0;
                        obj.learning_rate = 0.001; % Возвращаем начальный learning rate
                        fprintf('Применено случайное смещение параметров. Продолжение обучения...\n');
                    end
                end

                % Вывод прогресса
                fprintf('Эпоха %3d: Train MAE = %.4f | Val MAE = %.4f | LR = %.2e\n',...
                    epoch, train_error, test_error, obj.learning_rate);

                % Дополнительное условие выхода
                if obj.best_test_error < 0.5
                    break;
                end
            end
            obj.modelShell.SetAlfaBeta(obj.best_al, obj.best_bt);
        end
    end

    methods (Access = private)
        function Optimize(obj, XData, YData, BatchSize, Beta1, Beta2, Eps, VertexWeight, VertexImportance)
            arguments
                obj statModel.OptimizationMachine
                XData,
                YData,
                BatchSize,
                Beta1,
                Beta2,
                Eps,
                VertexWeight,
                VertexImportance
            end

            % --- Инициализация параметров Adam ---
            beta1 = Beta1;              % Для скользящего среднего градиента (момент)
            beta2 = Beta2;              % Для скользящего среднего квадратов градиента
            epsilon = Eps;              % Для численной стабильности
            batch_size = BatchSize;     % Размер мини-пакета
            num_vertex = obj.modelShell.GetNumOfVertex();

            % --- Инициализация моментов Adam ---
            if isempty(obj.m_al) % Первый вызов
                obj.m_al = zeros(num_vertex, num_vertex);
                obj.v_al = zeros(num_vertex, num_vertex);
                obj.m_bt = zeros(num_vertex, num_vertex);
                obj.v_bt = zeros(num_vertex, num_vertex);
                obj.t = 0;
            end

            % Индивидуальные коэффициенты для вершин
            vertex_weights = VertexWeight;  % Гиперпараметры
            vertex_lr_scale = VertexImportance; % Масштабирование LR для вершин

            % Регуляризация в зависимости от вершины
            vertex_reg = 0.01 * ones(num_vertex, 1) * VertexImportance; % Гиперпараметр

            % Инициализация массива ошибок
            obj.errorArray = zeros(size(XData,3), num_vertex);
            % Количество образцов для пакетной настройки
            num_samples = size(XData,3);
            % Количество батчей
            num_batches = ceil(num_samples / batch_size);

            % --- Цикл по батчам ---
            for batch_idx = 1:num_batches
                obj.t = obj.t + 1;

                % Определяем индексы
                batch_start = (batch_idx-1)*batch_size + 1;
                batch_end = min(batch_idx*batch_size, num_samples);
                batch_indices = batch_start:batch_end;
                num_in_batch = length(batch_indices);
               
                % Определяем матрицы для градиентов в пакете
                al_grad_batch = zeros(num_vertex, num_vertex);
                bt_grad_batch = zeros(num_vertex, num_vertex);
                
                % --- Цикл по сэмплам в пакете ---
                for k = 1:num_in_batch
                   
                    sample_idx = batch_indices(k);
                    xData = XData(:,:,sample_idx);
                    obj.modelShell.Forward(xData);

                    al = obj.modelShell.GetAlfas();
                    bt = obj.modelShell.GetBetas();

                    % Вычисление ошибки для каждой вершины
                    for i = 1:num_vertex
                        % Получаем ошибку для заданного i-го узла
                        error_i = obj.modelShell.GetResultByIndex(i) - YData(1,i,sample_idx);
                
                        % Весовая функция для важных вершин
                        weight = vertex_weights(i);
                        
                        % Градиенты с учетом индивидуальной важности вершины
                        if ~isnan(obj.modelShell.dFdAl(i))
                            al_grad_batch(i,:) = al_grad_batch(i,:) - weight * obj.modelShell.dFdAl(i) * error_i;
                        end

                        if ~isnan(obj.modelShell.dFdBt(i))
                            bt_grad_batch(i,:) = bt_grad_batch(i,:) - weight * obj.modelShell.dFdBt(i) * error_i;
                        end

                        al_grad_batch(i,i) = 0;
                        bt_grad_batch(i,i) = 0;
                    end
                end

           
                % --- Регуляризация с учетом вершины
                for i = 1:num_vertex
                    al_grad_batch(i,:) = al_grad_batch(i,:) + vertex_reg(i)*al(i,:);
                    bt_grad_batch(i,:) = bt_grad_batch(i,:) + vertex_reg(i)*bt(i,:);
                end

                % --- Защита от NaN/Inf ---
                al_grad_batch(isnan(al_grad_batch) | isinf(al_grad_batch)) = 0;
                bt_grad_batch(isnan(bt_grad_batch) | isinf(bt_grad_batch)) = 0;

                % Max-Min-клиппирование градиентов
                al_grad_batch = max(min(al_grad_batch, 5.0), -5.0);
                bt_grad_batch = max(min(bt_grad_batch, 5.0), -5.0);

                % --- Выполняем нормализацию градиента на размер пакета
                al_grad_batch = al_grad_batch / num_in_batch;
                bt_grad_batch = bt_grad_batch / num_in_batch;

                % --- Обновление моментов оптимизатора Adam ---
                obj.m_al = beta1 * obj.m_al + (1-beta1) * al_grad_batch;
                obj.m_bt = beta1 * obj.m_bt + (1-beta1) * bt_grad_batch;
                obj.v_al = beta2 * obj.v_al + (1-beta2) * (al_grad_batch.^2);
                obj.v_bt = beta2 * obj.v_bt + (1-beta2) * (bt_grad_batch.^2);

                % --- Коррекция смещений ---
                m_al_corr = obj.m_al / (1-beta1^obj.t);
                m_bt_corr = obj.m_bt / (1-beta1^obj.t);
                v_al_corr = obj.v_al / (1-beta2^obj.t);
                v_bt_corr = obj.v_bt / (1-beta2^obj.t);
              
                % --- Индивидуальное обновление параметров для вершин
                for i = 1:num_vertex
                    lr_scale = vertex_lr_scale(i);
                    al_update = obj.learning_rate * lr_scale * m_al_corr(i,:) ./ (sqrt(v_al_corr(i,:)) + epsilon);
                    bt_update = obj.learning_rate * lr_scale * m_bt_corr(i,:) ./ (sqrt(v_bt_corr(i,:)) + epsilon);

                    al(i,:) = al(i,:) + al_update;
                    bt(i,:) = bt(i,:) + bt_update;
                end

                % --- Обновляем параметры в модели ---
                obj.modelShell.SetAlfaBeta(al,bt);
            end
        end
    end
end