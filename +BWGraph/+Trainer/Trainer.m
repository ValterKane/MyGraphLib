classdef Trainer < handle

    properties (Access = private)
        % Кешированные данные о графе
        whiteNodeIndices
        blackNodeIndices
        incomingEdgesCache
        outgoingEdgesCache
        incomingNeighborsCache
        distancesCache

        % Кэш индексов для настройки
        NodeIndexMap

        % Моменты ADAM
        mAl
        vAl
        mBt
        vBt
        mGm
        vGm
        t
        % Параметры
        bestAl                     % Лучшие альфа-значения по результатам настройки
        bestBt                     % Лучшие бета-значения по результатам настройки
        bestGm                     % Лучшие gamma-значения по результатам настройки
        
        % Остальные параметры 
        graph                   BWGraph.GraphShell    % Графовая модель
        nodes
        errorArray                  % Массив ошибок обучения
        bestTestError = Inf;        % Поле для отслеживания улучшения ошибки
        minDelta = 0.0001;          % Минимальное улучшение для сохранения модели
        learningRate = 0.001;       % Начальный шаг обучения
        minLr = 1e-6;               % Минимальный шаг обучения
        lrReductionFactor = 0.5;    % Степень редуцирования шага обучения

        % Параметры настройки
        TrainingOptions BWGraph.Trainer.TrainingOptions
        
        % Параметры

        % --- Сохранение истории ошибок ---
        trainErrors = [];
        testErrors = [];
        % ---------------------------------

        % -- Параметры настройки на плато ---
        plateauCount = 0;         % Счетчик плато
        maxPlateauCount = 10;     % Максимальное количество плато перед остановкой
        randomShiftScale = 0.2;  % Масштаб случайного смещения параметров
        % -----------------------------------
    end

    methods (Access = public)
        function obj = Trainer(Graph, TrainingOptions)
            arguments
                Graph       BWGraph.GraphShell
                TrainingOptions  BWGraph.Trainer.TrainingOptions
            end

            obj.graph = Graph;
            obj.nodes = Graph.ListOfNodes;
            obj.TrainingOptions = TrainingOptions;

            % Инициализация структур для хранения лучших параметров
            nodes = obj.graph.ListOfNodes;
            numNodes = numel(nodes);

            % Создаем матрицы для хранения всех alfa и beta
            obj.bestAl = zeros(numNodes, numNodes);
            obj.bestBt = zeros(numNodes, numNodes);
            obj.bestGm = zeros(numNodes);

            % Заполняем начальными значениями
            for i = 1:numNodes
                node = nodes(i);
                edges = node.getOutEdges();
                for j = 1:numel(edges)
                    edge = edges(j);
                    targetId = edge.TargetNode.ID;
                    obj.bestAl(i, targetId) = edge.Alfa;
                    obj.bestBt(i, targetId) = edge.Beta;
                end
            end
        end

        function Train(obj,XDataTrain, YDataTrain, XDataTest, YDataTest)
            arguments
                obj             BWGraph.Trainer.Trainer
                XDataTrain      BWGraph.CustomMatrix.BWMatrix
                YDataTrain      BWGraph.CustomMatrix.BWMatrix
                XDataTest       BWGraph.CustomMatrix.BWMatrix
                YDataTest       BWGraph.CustomMatrix.BWMatrix
            end

            % Проверка индексов белых вершин
            allWhiteIndices = obj.graph.GetWhiteNodesIndices();
            if isempty(obj.TrainingOptions.TargetNodeIndices)
                obj.TrainingOptions.TargetNodeIndices = allWhiteIndices; % По умолчанию все белые вершины
            else
                % Проверяем, что все указанные индексы действительно являются белыми вершинами
                if ~all(ismember(obj.TrainingOptions.TargetNodeIndices, allWhiteIndices))
                    error('Указанные индексы должны соответствовать белым вершинам графа');
                end
            end

            % Копируем шаг обучения
            LearningRate = obj.TrainingOptions.LearningRate;

            % Проверка что входные данные - массивы BWMatrix
            if ~isa(XDataTrain, 'BWGraph.CustomMatrix.BWMatrix') || ...
                    ~isa(YDataTrain, 'BWGraph.CustomMatrix.BWMatrix') || ...
                    ~isa(XDataTest, 'BWGraph.CustomMatrix.BWMatrix') || ...
                    ~isa(YDataTest, 'BWGraph.CustomMatrix.BWMatrix')
                error('XData и YData должны быть массивами BWGraph.CustomMatrix.BWMatrix');
            end

            % Проверка размеров данных
            numWhiteNodes = obj.graph.GetNumOfWhiteNode;
            if YDataTrain(1).rowLength(1) ~= numWhiteNodes
                error("Размеры данных должны соответствовать количеству узлов в графе");
            end

            % Проверка согласованности данных
            if length(XDataTrain) ~= length(YDataTrain) || ...
                    length(XDataTest) ~= length(YDataTest)
                error("Количество примеров в XData и YData должно совпадать");
            end

            % Инициализация массива для хранения времени эпох
            epochTimes = zeros(1, obj.TrainingOptions.Epoches);
            errorDiffs = zeros(1, obj.TrainingOptions.Epoches);

            % Создаем фигуру для графиков
            figure('Name', 'Training Progress', 'NumberTitle', 'off', 'Position', [100 100 900 800]);
            % Создаем 4 субграфика в 2 колонки
            ax1 = subplot(2,2,1);  % Ошибки
            ax2 = subplot(2,2,2);  % Learning rate (логарифмическая шкала)
            ax3 = subplot(2,2,3);  % Время эпохи
            ax4 = subplot(2,2,4);  % Использование памяти

            fprintf('Старт процесса настройки. ЦФ=%s, Метрика=%s\n', obj.TrainingOptions.LossFunction, obj.TrainingOptions.ErrorMetric);

            for epoch = 1:obj.TrainingOptions.Epoches
                % Динамическое уменьшение LR
                if mod(epoch, 20) == 0
                    LearningRate = LearningRate / 2;
                end

                if epoch > 0.8*obj.TrainingOptions.Epoches
                    LearningRate = 0.005;
                end

                epochStart = tic;

                % Основной алгоритм настройки
                obj.Compute_V5(XDataTrain, YDataTrain);

                fprintf('\nНастройка на эпохе №%d завершена!\n',epoch)
                trainEerror = obj.trainErrors(end);

                % Расчет ошибки на тестовой выборке
                fprintf('\nВыполняю расчет метрики на тестовой выборке...\n')
                testError = obj.CalculateError(XDataTest, YDataTest, obj.TrainingOptions.TargetNodeIndices, obj.TrainingOptions.ErrorMetric);
                obj.testErrors(end+1) = testError;

                % Вычисление разницы между ошибками
                errorDiffs(epoch) = testError - trainEerror;
                
                % Проверка критериев остановки
                stopTraining = false;
                stopReason = '';
                
                % Проверка улучшения на тестовой выборке
                if testError < obj.bestTestError - obj.minDelta
                    obj.bestTestError = testError;
                    % Сохраняем лучшие параметры
                    obj.SaveBestParameters();
                end
                %     stopTraining = true;
                %     stopReason = ... 
                %         sprintf('Не обнаружено серьезного изменения в ошибке на тестовом множестве (%s = %3f',obj.TrainingOptions.ErrorMetric, testError);
                % end

                if epoch > 1
                    if errorDiffs(end-1) > 0 && errorDiffs(end) < 0 ...
                            || errorDiffs(end-1) < 0 && errorDiffs(end) >0
                        stopTraining = true;
                        stopReason = sprintf('Обнаружено переобучение (%s = %3f)',obj.TrainingOptions.ErrorMetric, testError);
                    end
                end

                % Критерий достижения целевой ошибки
                if obj.bestTestError < obj.TrainingOptions.TargetError
                    stopTraining = true;
                    stopReason = sprintf('Достигнута целевая ошибка (%s < %3f)',obj.TrainingOptions.ErrorMetric, obj.TrainingOptions.TargetError);
                end

                % Если сработал любой критерий остановки
                if stopTraining
                    fprintf('\nКритерий остановки: %s\n', stopReason);
                    fprintf('Обучение завершено на эпохе %d\n', epoch);
                    fprintf('Лучшая тестовая ошибка: %.4f\n', obj.bestTestError);
                    break;
                end

                % Замер времени эпохи и памяти после вычислений
                epochTimes(epoch) = toc(epochStart);

                fprintf('\nЭпоха %3d: Train %s = %.4f | Test %s = %.4f | LR = %.2e | Время = %.2f сек\n',...
                        epoch,obj.TrainingOptions.ErrorMetric, trainEerror, obj.TrainingOptions.ErrorMetric ,testError, LearningRate, epochTimes(epoch));

                % Обновление графиков
                % График ошибок
                plot(ax1, 1:epoch, obj.trainErrors, 'b-', 'LineWidth', 1.5);
                hold(ax1, 'on');
                plot(ax1, 1:epoch, obj.testErrors, 'r-', 'LineWidth', 1.5);

                hold(ax1, 'off');
                title(ax1, 'Ошибки настройки и тестирования');
                xlabel(ax1, 'Итерация');
                ylabel(ax1, sprintf('%s',obj.TrainingOptions.ErrorMetric));
                legend(ax1, {'Ошибка настройки', 'Ошибка тестирования'}, 'Location', 'best');
                grid(ax1, 'on');

                % График learning rate в логарифмической шкале
                semilogy(ax2, 1:epoch, obj.learningRate * ones(1, epoch), 'r.', 'MarkerSize', 10);
                hold(ax2, 'on');
                semilogy(ax2, 1:epoch, obj.learningRate * ones(1, epoch), 'r-', 'LineWidth', 0.5);
                hold(ax2, 'off');
                title(ax2, 'Шаг настройки (логарифмическая шкала)');
                xlabel(ax2, 'Итерация');
                ylabel(ax2, 'Шаг настройки (log)');
                grid(ax2, 'on');
                ylim(ax2, [obj.minLr/10, LearningRate*10]); % Динамические границы

                % График времени эпохи
                plot(ax3, 1:epoch, epochTimes(1:epoch), 'g-', 'LineWidth', 1.5);
                title(ax3, 'Время расчета эпохи');
                xlabel(ax3, 'Итерация');
                ylabel(ax3, 'Время (сек)');
                grid(ax3, 'on');

                % График разницы между ошибками
                plot(ax4, 1:epoch, errorDiffs(1:epoch), 'm-', 'LineWidth', 1.5);
                hold(ax4, 'on');
                % Линия нуля для reference
                plot(ax4, [1 epoch], [0 0], 'k--', 'LineWidth', 1);
                hold(ax4, 'off');
                title(ax4, 'Разница между между ошибками тестирования и настройки');
                xlabel(ax4, 'Итерация');
                ylabel(ax4, '\Delta E');
                legend(ax4, {'Разница ошибок', 'Нулевая линия'}, 'Location', 'best');
                grid(ax4, 'on');

                drawnow; % Обновляем графики

                % Восстанавливаем лучшие параметры
                obj.RestoreBestParameters();
            end

            % Финализация графиков
            sgtitle(sprintf('Обучение завершено на эпохе %d (Лучшая тестовая %s: %.4f)', epoch, obj.TrainingOptions.ErrorMetric, obj.bestTestError));
        end

        function graph = GetGraph(obj)
            graph = obj.graph;
        end
    end

    methods (Access = private)
        function SaveBestParameters(obj)
            % Сохраняет текущие параметры графа как лучшие
            obj.nodes = obj.graph.ListOfNodes;
            numNodes = numel(obj.nodes);

            % Очищаем предыдущие лучшие значения
            obj.bestAl = zeros(numNodes, numNodes);
            obj.bestBt = zeros(numNodes, numNodes);
            obj.bestGm = zeros(numNodes);

            % Сохраняем текущие значения
            for i = 1:numNodes
                node = obj.nodes(i);
                edges = node.getOutEdges();
                obj.bestGm(i) = node.Gamma;
                for j = 1:numel(edges)
                    edge = edges(j);
                    targetId = edge.TargetNode.ID;
                    obj.bestAl(i, targetId) = edge.Alfa;
                    obj.bestBt(i, targetId) = edge.Beta;
                end
            end
        end

        function RestoreBestParameters(obj)
            % Восстанавливает лучшие параметры в графе
            obj.nodes = obj.graph.ListOfNodes;
            for i = 1:numel(obj.nodes)
                node = obj.nodes(i);
                edges = node.getOutEdges();
                node.Gamma = obj.bestGm(i);
                for j = 1:numel(edges)
                    edge = edges(j);
                    targetId = edge.TargetNode.ID;
                    edge.Alfa = obj.bestAl(i, targetId);
                    edge.Beta = obj.bestBt(i, targetId);
                end
            end
        end

        function RandomShiftParameters(obj)
            % Применяет случайное смещение к параметрам графа
            obj.nodes = obj.graph.ListOfNodes;
            for i = 1:numel(obj.nodes)
                node = obj.nodes(i);
                edges = node.getOutEdges();
                for j = 1:numel(edges)
                    edge = edges(j);

                    % Генерируем случайные смещения
                    randomShiftAl = (rand() * 2 - 1) * obj.randomShiftScale;
                    randomShiftBt = (rand() * 2 - 1) * obj.randomShiftScale;

                    % Применяем смещения
                    edge.Alfa = edge.Alfa + randomShiftAl;
                    edge.Beta = edge.Beta + randomShiftBt;
                end
            end
        end

        function errorValue = CalculateError(obj, XData, YData, whiteNodeIndices, errorMetric)
            % Обновленный метод CalculateError с поддержкой:
            % - выбора белых вершин
            % - выбора метрики ошибки (MAE или MAPE)

            % Параметры по умолчанию
            if nargin < 4 || isempty(whiteNodeIndices)
                whiteNodeIndices = obj.graph.GetWhiteNodesIndices();
            end
            if nargin < 5
                errorMetric = 'mae'; % По умолчанию MAE
            end

            % Проверка допустимых значений метрики
            validMetrics = {'mae', 'mape'};
            if ~any(strcmpi(errorMetric, validMetrics))
                error('Недопустимая метрика ошибки. Допустимые значения: ''mae'', ''mape''');
            end

            % 1. Проверка входных данных
            if length(XData) ~= length(YData)
                error('Размеры XData и YData должны совпадать');
            end

            if isempty(whiteNodeIndices)
                errorValue = NaN;
                return;
            end

            % 2. Инициализация
            totalError = 0;
            totalPoints = 0;

            % 3. Основной цикл по примерам
            for i = 1:length(XData)
                % Проверка типов
                if ~isa(XData(i), 'BWGraph.CustomMatrix.BWMatrix') || ...
                        ~isa(YData(i), 'BWGraph.CustomMatrix.BWMatrix')
                    error('Неверный тип данных');
                end
    
                % Прямой проход
                obj.graph.Forward(XData(i));
                pred = obj.graph.GetModelResults();

                % Выбираем только указанные белые узлы
                selectedPred = pred(whiteNodeIndices);
                selectedTrue = YData(i).getRow(1);  % Получаем все значения белых узлов

                % Выбираем только указанные индексы
                allWhiteIndices = obj.graph.GetWhiteNodesIndices();
                [~, loc] = ismember(whiteNodeIndices, allWhiteIndices);
                selectedTrue = selectedTrue(loc);

                % Проверка размерности
                if length(selectedTrue) ~= length(whiteNodeIndices)
                    error('Несоответствие размеров в YData');
                end

                % Расчет ошибки в зависимости от выбранной метрики
                switch lower(errorMetric)
                    case 'mae'
                        % Средняя абсолютная ошибка
                        totalError = totalError + sum(abs(selectedPred - selectedTrue));
                    case 'mape'
                        % Средняя абсолютная процентная ошибка (с защитой от деления на 0)
                        epsilon = 1e-10; % Малое значение для избежания деления на 0
                        absTrue = abs(selectedTrue);
                        absTrue(absTrue < epsilon) = epsilon; % Заменяем нули на epsilon
                        totalError = totalError + sum(abs(selectedPred - selectedTrue) ./ absTrue);
                end
                totalPoints = totalPoints + length(whiteNodeIndices);

                obj.updateProgress(i, length(XData));
            end

            % 4. Финальное усреднение
            switch lower(errorMetric)
                case 'mae'
                    errorValue = totalError / totalPoints;
                case 'mape'
                    errorValue = (totalError / totalPoints) * 100; % В процентах
            end
        end

        
        function updateProgress(~, current, total, message)
            % Обновление прогресс-бара
            % current - текущая итерация
            % total - общее количество
            % message - дополнительное сообщение (опционально)

            persistent lastPercent;
            persistent lineLength;

            if isempty(lastPercent)
                lastPercent = -1;
                lineLength = 0;
            end

            percent = floor(current/total * 100);

            % Обновляем только если процент изменился
            if percent ~= lastPercent
                bars = floor(percent/5); % 20 символов = 100%

                if current == 1
                    % Первый вызов - выводим полную строку
                    fprintf('\nПрогресс: ');
                    lineLength = fprintf('[%s%s] %3d%%', ...
                        repmat('░', 1, 20), ...
                        repmat('░', 1, 0), ...
                        percent);

                    if nargin > 3 && ~isempty(message)
                        lineLength = lineLength + fprintf('  %s', message);
                    end

                else
                    % Возвращаем курсор к началу строки прогресса
                    if lineLength > 0
                        fprintf(repmat('\b', 1, lineLength));
                    end

                    % Выводим обновленный прогресс-бар
                    lineLength = fprintf('[%s%s] %3d%%', ...
                        repmat('█', 1, bars), ...
                        repmat('░', 1, 20-bars), ...
                        percent);

                    if nargin > 3 && ~isempty(message)
                        lineLength = lineLength + fprintf('  %s', message);
                    end
                end

                lastPercent = percent;

                % Завершение
                if current == total
                    fprintf('\n');
                    clear lastPercent;
                    clear lineLength;
                end
            end
        end

        
        function Compute_V5(obj, XData, YData)
            arguments
                obj                 BWGraph.Trainer.Trainer,
                XData
                YData
            end

            % --- Инициализация параметров ---
            epsilon = obj.TrainingOptions.Eps;
            numNodes = numel(obj.nodes);

            % Инициализация кешей
            if isempty(obj.whiteNodeIndices) || isempty(obj.blackNodeIndices)
                obj.whiteNodeIndices = obj.graph.GetWhiteNodesIndices();
                obj.blackNodeIndices = obj.graph.GetBlackNodesIndices();
            end

            if isempty(obj.incomingEdgesCache) || isempty(obj.outgoingEdgesCache) || isempty(obj.incomingNeighborsCache)
                obj.incomingEdgesCache = cell(numNodes, 1);
                obj.outgoingEdgesCache = cell(numNodes, 1);
                obj.incomingNeighborsCache = cell(numNodes, 1);

                for i = 1:numNodes
                    obj.incomingEdgesCache{i} = obj.graph.getIncomingEdges(obj.nodes(i));
                    obj.outgoingEdgesCache{i} = obj.nodes(i).getOutEdges();
                    obj.incomingNeighborsCache{i} = obj.graph.getIncomingNeighbors(obj.nodes(i));
                end
            end
            
            % Кэшируем индексы
            if isempty(obj.NodeIndexMap)
                obj.NodeIndexMap = containers.Map('KeyType', 'char', 'ValueType', 'double');
                for idx = 1:numNodes
                    obj.NodeIndexMap(class(obj.nodes(idx))) = idx;
                end
            end

            % Инициализация моментов ADAM
            if isempty(obj.mAl)
                obj.mAl = cell(numNodes, 1);
                obj.vAl = cell(numNodes, 1);
                obj.mBt = cell(numNodes, 1);
                obj.vBt = cell(numNodes, 1);
                obj.mGm = cell(numNodes, 1);
                obj.vGm = cell(numNodes, 1);
                numEdgesPerNode = cellfun(@numel, obj.outgoingEdgesCache);
                for i = 1:numNodes
                    obj.mAl{i} = zeros(1, numEdgesPerNode(i));
                    obj.vAl{i} = zeros(1, numEdgesPerNode(i));
                    obj.mBt{i} = zeros(1, numEdgesPerNode(i));
                    obj.vBt{i} = zeros(1, numEdgesPerNode(i));
                    obj.mGm{i} = 0;
                    obj.vGm{i} = 0;
                end
                obj.t = 0;
            end

            % --- Пакетная обработка ---
            numSamples = length(XData);
            total_errors = 0;
            total_points = 0;
            numBatches = ceil(numSamples / obj.TrainingOptions.BatchSize);

            for batchIdx = 1:numBatches
                obj.t = obj.t + 1;
                batchStart = (batchIdx-1)*obj.TrainingOptions.BatchSize + 1;
                batchEnd = min(batchIdx*obj.TrainingOptions.BatchSize, numSamples);
                batchIndices = batchStart:batchEnd;
                numInBatch = length(batchIndices);

                obj.updateProgress(batchIdx, numBatches);

                % Инициализация градиентов для всего батча
                batchAlGrad = cell(numNodes, 1);
                batchBtGrad = cell(numNodes, 1);
                batchGmGrad = cell(numNodes, 1);
                for i = 1:numNodes
                    numEdges = numel(obj.outgoingEdgesCache{i});
                    batchAlGrad{i} = zeros(1, numEdges);
                    batchBtGrad{i} = zeros(1, numEdges);
                    batchGmGrad{i} = 0;
                end

                % --- Обработка примеров в батче ---
                for k = 1:numInBatch
                    sampleIdx = batchIndices(k);
                    xMatrix = XData(sampleIdx);
                    yMatrix = YData(sampleIdx);

                    % Прямой проход (использует исправленный Forward)
                    modelValues = obj.graph.GetCurrentResult(xMatrix);
                    % Здесь пока берется только одна белая вершина
                    targetValues = yMatrix.getRow(1);

                    % После извлечения эталона увеличим количество на 1
                    total_points = total_points + 1;

                    % Вычисление ошибок для белых вершин
                    J_white = zeros(1, numNodes);

                    J_white(obj.whiteNodeIndices) = modelValues(obj.whiteNodeIndices) - targetValues;

                    % Расчет усредненной ошибки для целевых белых вершин
                    meanTargetError = mean(J_white(obj.TrainingOptions.TargetNodeIndices));

                    % Считаем метрику находу в процессе обучения
                    switch obj.TrainingOptions.ErrorMetric
                        case 'mae'
                            total_errors = total_errors + abs(meanTargetError);
                        case 'mse'
                            total_errors = total_errors + 0.5*(meanTargetError^2);
                        case 'rmse'
                            total_errors = total_errors + sqrt(0.5 * (meanTargetError^2));
                        otherwise
                            total_errors = total_errors + abs(meanTargetError);
                    end

                    % Cчитаем лосс-функцию
                    for i = obj.whiteNodeIndices
                        error = J_white(i);
                        switch obj.TrainingOptions.LossFunction
                            case 'mae'
                                % Mean Absolute Error
                                J_white(i) = sign(error) + obj.TrainingOptions.Lambda_Agg * sign(meanTargetError);
                            case 'mse'
                                % Mean Squared Error
                                J_white(i) = error + obj.TrainingOptions.Lambda_Agg * meanTargetError;
                            case 'huber'
                                % Huber Loss
                                delta = obj.TrainingOptions.HuberDelta;
                                if abs(error) <= delta
                                    % Квадратичная часть: L = 0.5 * error^2
                                    % Производная: error
                                    huber_deriv = error;
                                else
                                    % Линейная часть: L = delta * (|error| - 0.5*delta)
                                    % Производная: delta * sign(error)
                                    huber_deriv = delta * sign(error);
                                end

                                % Производная для meanTargetError
                                if abs(meanTargetError) <= delta
                                    target_deriv = meanTargetError;
                                else
                                    target_deriv = delta * sign(meanTargetError);
                                end
                                J_white(i) = huber_deriv + obj.TrainingOptions.Lambda_Agg * target_deriv;

                            case 'logcosh'
                                % Log-Cosh Loss: L = log(cosh(error))
                                % Производная: tanh(error)
                                J_white(i) = tanh(error) + obj.TrainingOptions.Lambda_Agg * tanh(meanTargetError);
                            otherwise
                                % Если что-то пошло не так, то MSE
                                J_white(i) = sign(error) + obj.TrainingOptions.Lambda_Agg * sign(meanTargetError);
                        end
                    end

                  
                    % --- Распространение ошибок для черных вершин по формуле (3.2) ---
                    % 1. Предварительно вычисляем коэффициенты δ для всех вершин
                    delta_in_cache = cell(numNodes, 1);
                    delta_out_cache = cell(numNodes, 1);

                    for i = 1:numNodes
                        incomingEdges = obj.incomingEdgesCache{i};
                        outgoingEdges = obj.outgoingEdgesCache{i};

                        % Вычисляем Σ α_e + 1 для исходящих рёбер
                        sum_alpha_out_plus_one = sum(arrayfun(@(e) e.Alfa + 1, outgoingEdges));

                        % Коэффициенты δ_In для входящих рёбер (формула 3.5)
                        delta_in = zeros(1, numNodes);
                        for edge_idx = 1:numel(incomingEdges)
                            e = incomingEdges(edge_idx);
                            sourceNode = e.SourceNode;
                            sourceIdx = obj.NodeIndexMap(class(sourceNode));
                            if ~isempty(sourceIdx)
                                delta_in(sourceIdx) = e.Alfa / sum_alpha_out_plus_one;
                            end
                        end
                        delta_in_cache{i} = delta_in;

                        % Коэффициенты δ_Out для исходящих рёбер (формула 3.4)
                        delta_out = zeros(1, numNodes);
                        for edge_idx = 1:numel(outgoingEdges)
                            e = outgoingEdges(edge_idx);
                            targetNode = e.TargetNode;
                            targetIdx = obj.NodeIndexMap(class(targetNode));
                            if ~isempty(targetIdx)
                                % Нужно вычислить знаменатель для целевой вершины
                                targetOutEdges = obj.outgoingEdgesCache{targetIdx};
                                sum_alpha_target = sum(arrayfun(@(edge) edge.Alfa, targetOutEdges));
                                delta_out(targetIdx) = e.Alfa / (sum_alpha_target + 1);
                            end
                        end
                        delta_out_cache{i} = delta_out;
                    end

                    J_total = zeros(1, numNodes);
                    J_total(obj.whiteNodeIndices) = J_white(obj.whiteNodeIndices);

                    A_in = eye(numNodes,numNodes);
                    A_out = eye(numNodes,numNodes);

                    max_iterations = length(obj.blackNodeIndices) + 1;

                    for iter = 1:max_iterations
                        J_prev = J_total;
                        J_new = J_total;
                        updated = false;

                        for i = obj.blackNodeIndices
                            delta_in = delta_in_cache{i};
                            delta_out = delta_out_cache{i};
                            incomingNeighbors = obj.incomingNeighborsCache{i};
                            outgoingEdges = obj.outgoingEdgesCache{i}';

                            % Сумма по входящим соседям
                            sum_in = 0;
                            for neighbor = incomingNeighbors
                                if ~isempty(neighbor)
                                    neighborIdx = obj.NodeIndexMap(class(neighbor));
                                    if ~isempty(neighborIdx) && delta_in(neighborIdx) ~= 0
                                        if A_in(i, neighborIdx) == 0 && J_prev(neighborIdx) ~= 0
                                            sum_in = sum_in + delta_in(neighborIdx) * J_prev(neighborIdx);
                                            A_in(i, neighborIdx) = 1;
                                        end
                                    end
                                end
                            end

                            % Сумма по исходящим соседям
                            sum_out = 0;
                            for e = outgoingEdges
                                targetNode = e.TargetNode;
                                targetIdx = obj.NodeIndexMap(class(targetNode));
                                if ~isempty(targetIdx) && delta_out(targetIdx) ~= 0
                                    if A_out(i, targetIdx) == 0 && J_prev(targetIdx) ~= 0
                                        sum_out = sum_out + delta_out(targetIdx) * J_prev(targetIdx);
                                        A_out(i, targetIdx) = 1;
                                    end
                                end
                            end

                            new_value = sum_in + sum_out;
                            if new_value ~= 0 && new_value ~= J_new(i)
                                J_new(i) = new_value;
                                updated = true;
                            end
                        end

                        J_total = J_new;

                        if ~updated
                            break;
                        end
                    end

                    % Вычисляем все производные в топологическом порядке
                    [alpha_derivatives, beta_derivatives, gamma_derivatives] = obj.graph.computeAllDerivativesInOrder(xMatrix);

                    % --- Вычисление градиентов ---
                    % Используем J_total для вычисления производных
                    for i = 1:numNodes
                        if J_total(i) == 0, continue; end
                        edges = obj.outgoingEdgesCache{i};
                        
                        % Градиенты для вершины (gamma)
                        key_gamma = sprintf('node%d_gamma', i);
                        dF_dgamma = gamma_derivatives(key_gamma);
                        batchGmGrad{i} = batchGmGrad{i} - dF_dgamma * J_total(i);
                        
                        % Градиенты для исходящих рёбер (alpha и beta)
                        for j = 1:numel(edges)
                            edge = edges(j);

                            % Градиент для α
                            key_alpha = sprintf('node%d_edge%d_alpha_out', i, edge.ID);
                            if isKey(alpha_derivatives, key_alpha)
                                dF_dalpha_out = alpha_derivatives(key_alpha);
                            end

                            % Градиент для β
                            key_beta = sprintf('node%d_edge%d_beta_out', i, edge.ID);
                            if isKey(beta_derivatives, key_beta)
                                dF_dbeta_out = beta_derivatives(key_beta);
                            end

                            % Обновление градиентов с регуляризацией (формула 3.10-3.11)
                            batchAlGrad{i}(j) = batchAlGrad{i}(j) - dF_dalpha_out * J_total(i);
                            batchBtGrad{i}(j) = batchBtGrad{i}(j) - dF_dbeta_out * J_total(i);
                        end

                        % Градиенты для входящих рёбер (alpha и beta)
                        incomingEdges = obj.incomingEdgesCache{i};
                        for j = 1:numel(incomingEdges)
                            edge = incomingEdges(j);
                            sourceNode = edge.SourceNode;
                            sourceIdx = obj.NodeIndexMap(class(sourceNode));

                            if isempty(sourceIdx), continue; end
                            % Градиент для α
                            key_alpha = sprintf('node%d_edge%d_alpha_in', i, edge.ID);
                            if isKey(alpha_derivatives, key_alpha)
                                dF_dalpha_in = alpha_derivatives(key_alpha);
                            end

                            % Градиент для β
                            key_beta = sprintf('node%d_edge%d_beta_in', i, edge.ID);
                            if isKey(beta_derivatives, key_beta)
                                dF_dbeta_in = beta_derivatives(key_beta);
                            end

                            if ~isnan(dF_dalpha_in) && ~isnan(dF_dbeta_in)
                                % Находим позицию этого ребра в исходящих ребрах sourceNode
                                sourceEdges = obj.outgoingEdgesCache{sourceIdx};
                                edgePos = find(sourceEdges == edge, 1);

                                if ~isempty(edgePos)
                                    batchAlGrad{sourceIdx}(edgePos) = batchAlGrad{sourceIdx}(edgePos) - ...
                                        dF_dalpha_in * J_total(i);
                                    batchBtGrad{sourceIdx}(edgePos) = batchBtGrad{sourceIdx}(edgePos) - ...
                                        dF_dbeta_in * J_total(i);
                                end
                            end
                        end
                    end

                    % Добавляем L2 регуляризацию (λ₁ в формуле 3.7)
                    for i = 1:numNodes
                        % Регуляризация по gamma
                        batchGmGrad{i} = batchGmGrad{i} + obj.TrainingOptions.Lambda_Gamma * obj.nodes(i).Gamma;
                        edges = obj.outgoingEdgesCache{i};
                        for j = 1:numel(edges)
                            edge = edges(j);
                            batchAlGrad{i}(j) = batchAlGrad{i}(j) + obj.TrainingOptions.Lambda_Alph * edge.Alfa;
                            batchBtGrad{i}(j) = batchBtGrad{i}(j) + obj.TrainingOptions.Lambda_Beta * edge.Beta;
                        end
                    end
                end

                % --- Нормализация и обрезка градиентов ---
                invNumInBatch = 1 / numInBatch;
                for i = 1:numNodes
                    if ~isempty(batchAlGrad{i})
                        batchAlGrad{i} = min(max(batchAlGrad{i} * invNumInBatch, obj.TrainingOptions.ClipDown), obj.TrainingOptions.ClipUp);
                    end
                    if ~isempty(batchBtGrad{i})
                        batchBtGrad{i} = min(max(batchBtGrad{i} * invNumInBatch, obj.TrainingOptions.ClipDown), obj.TrainingOptions.ClipUp);
                    end
                    if ~isempty(batchGmGrad{i})
                        batchGmGrad{i} = min(max(batchGmGrad{i} * invNumInBatch, obj.TrainingOptions.ClipDown), obj.TrainingOptions.ClipUp);
                    end
                end

                % --- Обновление параметров с помощью ADAM ---
                beta1_t = obj.TrainingOptions.Beta1^obj.t;
                beta2_t = obj.TrainingOptions.Beta2^obj.t;
                mCorrFactor = 1 / (1 - beta1_t);
                vCorrFactor = 1 / (1 - beta2_t);

                for i = 1:numNodes
                    edges = obj.outgoingEdgesCache{i};
                    if isempty(edges), continue; end

                    % Обновление моментов ADAM (формулы 3.14-3.15)
                    obj.mAl{i} = obj.TrainingOptions.Beta1 * obj.mAl{i} + (1-obj.TrainingOptions.Beta1) * batchAlGrad{i};
                    obj.vAl{i} = obj.TrainingOptions.Beta2 * obj.vAl{i} + (1-obj.TrainingOptions.Beta2) * (batchAlGrad{i}.^2);
                    obj.mBt{i} = obj.TrainingOptions.Beta1 * obj.mBt{i} + (1-obj.TrainingOptions.Beta1) * batchBtGrad{i};
                    obj.vBt{i} = obj.TrainingOptions.Beta2 * obj.vBt{i} + (1-obj.TrainingOptions.Beta2) * (batchBtGrad{i}.^2);
                    obj.mGm{i} = obj.TrainingOptions.Beta1 * obj.mGm{i} + (1-obj.TrainingOptions.Beta1) * batchGmGrad{i};
                    obj.vGm{i} = obj.TrainingOptions.Beta2 * obj.vGm{i} + (1-obj.TrainingOptions.Beta2) * (batchGmGrad{i}.^2);

                    % Применение обновлений (формула 3.16)
                    lr = obj.learningRate * obj.TrainingOptions.NodeSize(i);
                    sqrtVAl = sqrt(obj.vAl{i} * vCorrFactor) + epsilon;
                    sqrtVBt = sqrt(obj.vBt{i} * vCorrFactor) + epsilon;
                    sqrtVGm = sqrt(obj.vGm{i} * vCorrFactor) + epsilon;

                    alfaUpdates = lr * (obj.mAl{i} * mCorrFactor) ./ sqrtVAl;
                    betaUpdates = lr * (obj.mBt{i} * mCorrFactor) ./ sqrtVBt;
                    obj.nodes(i).Gamma = obj.nodes(i).Gamma + lr * (obj.mGm{i} * mCorrFactor) / sqrtVGm;

                    for j = 1:numel(edges)
                        edges(j).Alfa = edges(j).Alfa + alfaUpdates(j);
                        edges(j).Beta = edges(j).Beta + betaUpdates(j);
                    end
                end
            end
            obj.trainErrors(end+1) = total_errors / total_points;
        end
    end
end

