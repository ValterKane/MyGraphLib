classdef NonLinearTrainer < handle

    properties (Access = private)
        % Кешированные данные о графе
        whiteNodeIndices
        blackNodeIndices
        incomingEdgesCache
        outgoingEdgesCache
        sourceIndicesCache
        incomingNeighborsCache
        targetIndicesCache
        distancesCache
        edgePositionsCache

        % Моменты ADAM для всех четырёх параметров
        mAl
        vAl
        mBt
        vBt
        mGm
        vGm
        mDl
        vDl
        t

        % Остальные параметры
        epochsNoImprove = 0;      % Фактическое количество эпох без улучшения
        bestAl                     % Лучшие альфа-значения по результатам настройки
        bestBt                     % Лучшие бета-значения по результатам настройки
        bestGm                     % Лучшие гамма-значения по результатам настройки
        bestDl                     % Лучшие дельта-значения по результатам настройки
    end

    properties
        graph BWGraph.NonLinearBWGraph.GraphShellNonlinear
        errorArray                  % Массив ошибок обучения
        bestTestError = Inf;        % Поле для отслеживания улучшения ошибки
        patience = 25;              % Количество эпох без улучшения для ранней остановки
        minDelta = 0.0001;          % Минимальное улучшение для сохранения модели
        learningRate = 0.001;       % Начальный шаг обучения
        minLr = 1e-6;               % Минимальный шаг обучения
        lrReductionFactor = 0.5;    % Степень редуцирования шага обучения
        numEpoch                    % Количество эпох обучения
        BatchSize = 32;

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
        function obj = NonLinearTrainer(graph, BatchSize)
            arguments
                graph       BWGraph.NonLinearBWGraph.GraphShellNonlinear
                BatchSize   {mustBePositive, mustBeFinite},
            end
            obj.graph = graph;
            % Инициализация лучших параметров
            obj.InitializeBestParameters();
            obj.BatchSize = BatchSize;
        end

        function Train(obj, XDataTrain, YDataTrain, XDataTest, YDataTest, ...
                LearningRate, Beta1, Beta2, Eps, NodeWeight, NodeSize, Epoches, ...
                ClipUp, ClipDown, TargetError, Lambda, Lambda_Agg, targetNodeIndices, errorMetric)
            arguments
                obj             BWGraph.NonLinearBWGraph.Trainer.NonLinearTrainer
                XDataTrain
                YDataTrain
                XDataTest
                YDataTest
                LearningRate    {mustBePositive, mustBeFinite}
                Beta1           {mustBePositive, mustBeFinite}
                Beta2           {mustBePositive, mustBeFinite}
                Eps             {mustBePositive, mustBeFinite}
                NodeWeight      (1,:)
                NodeSize        (1,:)
                Epoches         {mustBePositive, mustBeInteger}
                ClipUp          {mustBeFinite}
                ClipDown        {mustBeFinite}
                TargetError     {mustBePositive, mustBeFinite}
                Lambda          {mustBePositive}
                Lambda_Agg          {mustBeNonnegative}
                targetNodeIndices = [] % По умолчанию анализируются все белые вершины
                errorMetric = 'mae'
            end

            % Проверка допустимых значений метрики
            validMetrics = {'mae', 'mape'};
            if ~any(strcmpi(errorMetric, validMetrics))
                error('Недопустимая метрика ошибки. Допустимые значения: ''mae'', ''mape''');
            end

            % Проверка индексов белых вершин
            allWhiteIndices = obj.graph.GetWhiteNodesIndices();
            if isempty(targetNodeIndices)
                targetNodeIndices = allWhiteIndices; % По умолчанию все белые вершины
            else
                % Проверяем, что все указанные индексы действительно являются белыми вершинами
                if ~all(ismember(targetNodeIndices, allWhiteIndices))
                    error('Указанные индексы должны соответствовать белым вершинам графа');
                end
            end

            % Проверка что входные данные - массивы BWMatrix
            if ~isa(XDataTrain, 'BWGraph.CustomMatrix.BWMatrix') || ...
                    ~isa(YDataTrain, 'BWGraph.CustomMatrix.BWMatrix') || ...
                    ~isa(XDataTest, 'BWGraph.CustomMatrix.BWMatrix') || ...
                    ~isa(YDataTest, 'BWGraph.CustomMatrix.BWMatrix')
                error('XData и YData должны быть массивами BWGraph.CustomMatrix.BWMatrix');
            end

            % Проверка размеров данных
            numWhiteNodes = obj.graph.numOfWhiteNodes;
            if YDataTrain(1).rowLength(1) ~= numWhiteNodes
                error("Размеры данных должны соответствовать количеству узлов в графе");
            end

            % Проверка согласованности данных
            if length(XDataTrain) ~= length(YDataTrain) || ...
                    length(XDataTest) ~= length(YDataTest)
                error("Количество примеров в XData и YData должно совпадать");
            end

            % -- Применение настроек модели
            obj.numEpoch = Epoches;
            obj.learningRate = LearningRate;

            % Инициализация массива для хранения времени эпох
            epochTimes = zeros(1, Epoches);
            errorDiffs = zeros(1, Epoches);
            diffIncreaseCount = 0;

            % Массив для хранения эпох, когда происходило смещение
            shiftEpochs = [];

            % Создаем фигуру для графиков
            figure('Name', 'Training Progress', 'NumberTitle', 'off', 'Position', [100 100 900 800]);
            % Создаем 4 субграфика в 2 колонках
            ax1 = subplot(2,2,1);  % Ошибки
            ax2 = subplot(2,2,2);  % Learning rate (логарифмическая шкала)
            ax3 = subplot(2,2,3);  % Время эпохи
            ax4 = subplot(2,2,4);  % Использование памяти

            for epoch = 1:obj.numEpoch
                % Динамическое уменьшение LR
                if mod(epoch, 100) == 0
                    obj.learningRate = max(obj.minLr, LearningRate / sqrt(epoch));
                end

                epochStart = tic;

                % Обучение на обучающей выборке
                obj.Compute_V5_Nonlinear(XDataTrain, YDataTrain, ...
                    Beta1, Beta2, Eps, NodeWeight, NodeSize, ClipUp, ClipDown, Lambda, Lambda_Agg, targetNodeIndices);

                % Расчет ошибки на обучающей выборке
                trainError = obj.CalculateError(XDataTrain, YDataTrain, targetNodeIndices, errorMetric);
                obj.trainErrors(end+1) = trainError;

                % Расчет ошибки на тестовой выборке
                testError = obj.CalculateError(XDataTest, YDataTest, targetNodeIndices, errorMetric);
                obj.testErrors(end+1) = testError;

                % Вычисление разницы между ошибками
                errorDiffs(epoch) = testError - trainError;

                % Проверка улучшения на тестовой выборке
                if testError < obj.bestTestError - obj.minDelta
                    obj.bestTestError = testError;
                    obj.epochsNoImprove = 0;
                    % Сохраняем лучшие параметры
                    obj.SaveBestParameters();
                else
                    obj.epochsNoImprove = obj.epochsNoImprove + 1;
                end

                % Уменьшение LR при застое
                if obj.epochsNoImprove > 0 && mod(obj.epochsNoImprove, 5) == 0
                    obj.learningRate = max(obj.minLr, LearningRate / sqrt(epoch));
                    fprintf('Уменьшение LR до %.2e на эпохе %d\n', obj.learningRate, epoch);
                end

                % Проверка критериев остановки
                stopTraining = false;
                stopReason = '';

                % Критерий плато
                if obj.epochsNoImprove >= obj.patience
                    obj.plateauCount = obj.plateauCount + 1;
                    fprintf('Обнаружено плато ошибки (попытка %d из %d) на эпохе %d\n', ...
                        obj.plateauCount, obj.maxPlateauCount, epoch);

                    if obj.plateauCount >= obj.maxPlateauCount
                        stopTraining = true;
                        stopReason = 'Достигнуто максимальное количество плато';
                    else
                        % Случайное смещение параметров (всех четырёх)
                        obj.RandomShiftParametersAll();
                        % Запоминаем эпоху смещения
                        shiftEpochs(end+1) = epoch;
                        % Сброс счетчиков и learning rate
                        obj.epochsNoImprove = 0;
                        diffIncreaseCount = 0;
                        obj.learningRate = LearningRate;
                        fprintf('Применено случайное смещение параметров. Продолжение обучения...\n');
                    end
                end

                % Критерий достижения целевой ошибки
                if obj.bestTestError < TargetError
                    stopTraining = true;
                    stopReason = sprintf('Достигнута целевая ошибка (MAE < %.3f)', TargetError);
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

                if errorMetric == "mae"
                    % Вывод прогресса
                    fprintf('Эпоха %3d: Train MAE = %.4f | Test MAE = %.4f | LR = %.2e | Время = %.2f сек\n',...
                        epoch, trainError, testError, obj.learningRate, epochTimes(epoch));
                else
                    % Вывод прогресса
                    fprintf('Эпоха %3d: Train MAPE = %.4f | Test MAPE = %.4f | LR = %.2e | Время = %.2f сек\n',...
                        epoch, trainError, testError, obj.learningRate, epochTimes(epoch));
                end

                % Обновление графиков
                % График ошибок
                plot(ax1, 1:epoch, obj.trainErrors, 'b-', 'LineWidth', 1.5);
                hold(ax1, 'on');
                plot(ax1, 1:epoch, obj.testErrors, 'r-', 'LineWidth', 1.5);

                % Добавляем вертикальные линии для эпох со смещением
                if ~isempty(shiftEpochs)
                    yLimits = ylim(ax1);
                    for i = 1:length(shiftEpochs)
                        plot(ax1, [shiftEpochs(i) shiftEpochs(i)], yLimits, 'k--', 'LineWidth', 1);
                    end
                end

                hold(ax1, 'off');
                title(ax1, 'Ошибки настройки и тестирования');
                xlabel(ax1, 'Итерация');
                ylabel(ax1, 'MAE');
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
            sgtitle(sprintf('Обучение завершено на эпохе %d (Лучшая тестовая MAE: %.4f)', epoch, obj.bestTestError));
        end

        function graph = GetGraph(obj)
            graph = obj.graph;
        end
    end

    methods (Access = private)
        function InitializeBestParameters(obj)
            % Инициализация структур для хранения лучших параметров
            nodes = obj.graph.ListOfNodes;
            numNodes = numel(nodes);

            % Создаем матрицы для хранения всех параметров
            obj.bestAl = zeros(numNodes, numNodes);
            obj.bestBt = zeros(numNodes, numNodes);
            obj.bestGm = zeros(numNodes, numNodes);
            obj.bestDl = zeros(numNodes, numNodes);

            % Заполняем начальными значениями
            for i = 1:numNodes
                node = nodes(i);
                edges = node.getOutEdges();
                for j = 1:numel(edges)
                    edge = edges(j);
                    targetId = edge.TargetNode.ID;
                    obj.bestAl(i, targetId) = edge.Alfa;
                    obj.bestBt(i, targetId) = edge.Beta;
                    obj.bestGm(i, targetId) = edge.Gamma;
                    obj.bestDl(i, targetId) = edge.Delta;
                end
            end
        end

        function SaveBestParameters(obj)
            % Сохраняет текущие параметры графа как лучшие
            nodes = obj.graph.ListOfNodes;
            numNodes = numel(nodes);

            % Очищаем предыдущие лучшие значения
            obj.bestAl = zeros(numNodes, numNodes);
            obj.bestBt = zeros(numNodes, numNodes);
            obj.bestGm = zeros(numNodes, numNodes);
            obj.bestDl = zeros(numNodes, numNodes);

            % Сохраняем текущие значения
            for i = 1:numNodes
                node = nodes(i);
                edges = node.getOutEdges();
                for j = 1:numel(edges)
                    edge = edges(j);
                    targetId = edge.TargetNode.ID;
                    obj.bestAl(i, targetId) = edge.Alfa;
                    obj.bestBt(i, targetId) = edge.Beta;
                    obj.bestGm(i, targetId) = edge.Gamma;
                    obj.bestDl(i, targetId) = edge.Delta;
                end
            end
        end

        function RestoreBestParameters(obj)
            % Восстанавливает лучшие параметры в графе
            nodes = obj.graph.ListOfNodes;
            for i = 1:numel(nodes)
                node = nodes(i);
                edges = node.getOutEdges();
                for j = 1:numel(edges)
                    edge = edges(j);
                    targetId = edge.TargetNode.ID;
                    edge.Alfa = obj.bestAl(i, targetId);
                    edge.Beta = obj.bestBt(i, targetId);
                    edge.Gamma = obj.bestGm(i, targetId);
                    edge.Delta = obj.bestDl(i, targetId);
                end
            end
        end

        function RandomShiftParametersAll(obj)
            % Применяет случайное смещение ко всем четырём параметрам графа
            nodes = obj.graph.ListOfNodes;
            for i = 1:numel(nodes)
                node = nodes(i);
                edges = node.getOutEdges();
                for j = 1:numel(edges)
                    edge = edges(j);

                    % Генерируем случайные смещения для всех параметров
                    randomShiftAl = (rand() * 2 - 1) * obj.randomShiftScale;
                    randomShiftBt = (rand() * 2 - 1) * obj.randomShiftScale;
                    randomShiftGm = (rand() * 2 - 1) * obj.randomShiftScale;
                    randomShiftDl = (rand() * 2 - 1) * obj.randomShiftScale;

                    % Применяем смещения
                    edge.Alfa = edge.Alfa + randomShiftAl;
                    edge.Beta = edge.Beta + randomShiftBt;
                    edge.Gamma = edge.Gamma + randomShiftGm;
                    edge.Delta = edge.Delta + randomShiftDl;
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

                % Получаем результат
                pred = obj.graph.GetCurrentResult(XData(i));

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
            end

            % 4. Финальное усреднение
            switch lower(errorMetric)
                case 'mae'
                    errorValue = totalError / totalPoints;
                case 'mape'
                    errorValue = (totalError / totalPoints) * 100; % В процентах
            end
        end

        function Compute_V5_Nonlinear(obj, XData, YData, Beta1, Beta2, Eps, NodesWeights, NodesLRScale, ClipUp, ClipDown, Lambda, lambda_Agg, targetWhiteIndices)
    arguments
        obj                 BWGraph.NonLinearBWGraph.Trainer.NonLinearTrainer,
        XData
        YData
        Beta1               {mustBePositive, mustBeFinite},
        Beta2               {mustBePositive, mustBeFinite},
        Eps                 {mustBePositive, mustBeFinite},
        NodesWeights        (1,:)      % Веса вершин для степени настройки параметров
        NodesLRScale       (1,:)      % Масштабирование LR для вершин
        ClipUp             {mustBeFinite},
        ClipDown           {mustBeFinite},
        Lambda             {mustBePositive},
        lambda_Agg         {mustBeNonnegative}, % Коэффициент влияния усредненной ошибки
        targetWhiteIndices (1,:)      % Индексы целевых белых вершин
    end

    % --- Инициализация параметров ---
    beta1 = Beta1;
    beta2 = Beta2;
    epsilon = Eps;
    nodes = obj.graph.ListOfNodes;
    numNodes = numel(nodes);
    
    % Проверка типа графа
    if ~isa(obj.graph, 'BWGraph.NonLinearBWGraph.GraphShellNonlinear')
        error('Граф должен быть типа GraphShellNonlinear для нелинейной модели');
    end

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
            obj.incomingEdgesCache{i} = obj.graph.getIncomingEdges(nodes(i));
            obj.outgoingEdgesCache{i} = nodes(i).getOutEdges();
            obj.incomingNeighborsCache{i} = obj.graph.getIncomingNeighbors(nodes(i));
        end
    end

    % Инициализация моментов ADAM для всех 4 параметров
    if isempty(obj.mAl)
        obj.mAl = cell(numNodes, 1);
        obj.vAl = cell(numNodes, 1);
        obj.mBt = cell(numNodes, 1);
        obj.vBt = cell(numNodes, 1);
        obj.mGm = cell(numNodes, 1);
        obj.vGm = cell(numNodes, 1);
        obj.mDl = cell(numNodes, 1);
        obj.vDl = cell(numNodes, 1);
        
        numEdgesPerNode = cellfun(@numel, obj.outgoingEdgesCache);
        for i = 1:numNodes
            obj.mAl{i} = zeros(1, numEdgesPerNode(i));
            obj.vAl{i} = zeros(1, numEdgesPerNode(i));
            obj.mBt{i} = zeros(1, numEdgesPerNode(i));
            obj.vBt{i} = zeros(1, numEdgesPerNode(i));
            obj.mGm{i} = zeros(1, numEdgesPerNode(i));
            obj.vGm{i} = zeros(1, numEdgesPerNode(i));
            obj.mDl{i} = zeros(1, numEdgesPerNode(i));
            obj.vDl{i} = zeros(1, numEdgesPerNode(i));
        end
        obj.t = 0;
    end

    % --- Пакетная обработка ---
    numSamples = length(XData);
    numBatches = ceil(numSamples / obj.BatchSize);

    for batchIdx = 1:numBatches
        obj.t = obj.t + 1;
        batchStart = (batchIdx-1)*obj.BatchSize + 1;
        batchEnd = min(batchIdx*obj.BatchSize, numSamples);
        batchIndices = batchStart:batchEnd;
        numInBatch = length(batchIndices);

        % Инициализация градиентов для всего батча для 4 параметров
        batchAlGrad = cell(numNodes, 1);
        batchBtGrad = cell(numNodes, 1);
        batchGmGrad = cell(numNodes, 1);
        batchDlGrad = cell(numNodes, 1);
        
        for i = 1:numNodes
            numEdges = numel(obj.outgoingEdgesCache{i});
            batchAlGrad{i} = zeros(1, numEdges);
            batchBtGrad{i} = zeros(1, numEdges);
            batchGmGrad{i} = zeros(1, numEdges);
            batchDlGrad{i} = zeros(1, numEdges);
        end

        % --- Обработка примеров в батче ---
        validSamples = 0; % Счетчик валидных примеров
        
        for k = 1:numInBatch
            sampleIdx = batchIndices(k);
            xMatrix = XData(sampleIdx);
            yMatrix = YData(sampleIdx);

            try
                % Прямой проход с нелинейной моделью
                [modelValues, converged] = obj.graph.ForwardNonlinear(xMatrix, 200, 1e-8); % Увеличиваем max_iterations
                
                if ~converged
                    % Если не сошлось, пробуем с меньшим tolerance
                    [modelValues, converged] = obj.graph.ForwardNonlinear(xMatrix, 500, 1e-6);
                end
                
                if ~converged
                    warning('Прямой проход не сошелся для примера %d', sampleIdx);
                    continue; % Пропускаем этот пример
                end
                
                % Проверка на конечные значения
                if any(~isfinite(modelValues))
                    warning('Нефинитные значения в прямом проходе для примера %d', sampleIdx);
                    continue; % Пропускаем этот пример
                end
                
                % Проверка на большие значения
                if any(abs(modelValues) > 1e10)
                    warning('Слишком большие значения в прямом проходе для примера %d', sampleIdx);
                    continue; % Пропускаем этот пример
                end
                
                targetValues = yMatrix.getRow(1);
                
                % Проверка целевых значений
                if any(~isfinite(targetValues))
                    warning('Нефинитные целевые значения для примера %d', sampleIdx);
                    continue;
                end

                % Вычисление ошибок для белых вершин (MAE)
                whiteErrors = zeros(1, numNodes);
                whiteIndices = obj.whiteNodeIndices;
                
                if ~isempty(whiteIndices)
                    % Проверяем, что индексы в пределах диапазона
                    validWhiteIndices = whiteIndices(whiteIndices <= length(modelValues) & ...
                                                     whiteIndices <= length(targetValues));
                    
                    if ~isempty(validWhiteIndices)
                        whiteErrors(validWhiteIndices) = modelValues(validWhiteIndices) - ...
                                                         targetValues(validWhiteIndices);
                    end
                end

                % Проверка целевых белых вершин
                validTargetWhiteIndices = targetWhiteIndices(...
                    targetWhiteIndices <= length(whiteErrors) & ...
                    targetWhiteIndices <= numNodes);
                
                if isempty(validTargetWhiteIndices)
                    continue;
                end

                % Расчет усредненной ошибки для целевых белых вершин
                targetErrors = whiteErrors(validTargetWhiteIndices);
                
                % Исключаем нулевые ошибки из среднего
                nonZeroErrors = targetErrors(targetErrors ~= 0);
                if ~isempty(nonZeroErrors)
                    meanTargetError = mean(nonZeroErrors);
                else
                    meanTargetError = 0;
                end

                % Добавление усредненной ошибки к белым вершинам
                J_white = zeros(1, numNodes);
                for i = validWhiteIndices
                    J_white(i) = 0.5 * (whiteErrors(i)^2) + lambda_Agg * meanTargetError;
                end

                % --- Распространение ошибок для черных вершин ---
                J_black = zeros(1, numNodes);
                J_total = zeros(1, numNodes);
                J_total(validWhiteIndices) = J_white(validWhiteIndices);

                % Только если есть белые вершины
                if ~isempty(validWhiteIndices)
                    % 1. Предварительно вычисляем коэффициенты δ_in и δ_out для всех вершин
                    delta_in_cache = cell(numNodes, 1);
                    delta_out_cache = cell(numNodes, 1);

                    for i = 1:numNodes
                        if i > numNodes, continue; end
                        
                        outgoingEdges = obj.outgoingEdgesCache{i};
                        if isempty(outgoingEdges)
                            delta_in_cache{i} = zeros(1, numNodes);
                            delta_out_cache{i} = zeros(1, numNodes);
                            continue;
                        end

                        % Для нелинейной модели знаменатель: S_out - Γ
                        sum_alpha_out_plus_one = 0;
                        for edge_idx = 1:numel(outgoingEdges)
                            e = outgoingEdges(edge_idx);
                            if isfinite(e.Alfa)
                                sum_alpha_out_plus_one = sum_alpha_out_plus_one + e.Alfa;
                            end
                        end
                        sum_alpha_out_plus_one = sum_alpha_out_plus_one + 1;
                        
                        % Вычисляем Γ = Σδ_e*F_u
                        Gamma = 0;
                        for edge_idx = 1:numel(outgoingEdges)
                            e = outgoingEdges(edge_idx);
                            targetNode = e.TargetNode;
                            targetIdx = find(nodes == targetNode, 1);
                            if ~isempty(targetIdx) && targetIdx <= length(modelValues)
                                F_u = modelValues(targetIdx);
                                if isprop(e, 'Delta') && isfinite(e.Delta) && isfinite(F_u)
                                    Gamma = Gamma + e.Delta * F_u;
                                end
                            end
                        end
                        
                        denominator = sum_alpha_out_plus_one - Gamma;
                        
                        % Защита от деления на ноль или очень маленького знаменателя
                        if abs(denominator) < 1e-10
                            denominator = sign(denominator) * 1e-10;
                        end

                        % Коэффициенты δ_In для входящих рёбер
                        delta_in = zeros(1, numNodes);
                        incomingEdges = obj.incomingEdgesCache{i};
                        for edge_idx = 1:numel(incomingEdges)
                            e = incomingEdges(edge_idx);
                            sourceNode = e.SourceNode;
                            sourceIdx = find(nodes == sourceNode, 1);
                            if ~isempty(sourceIdx) && sourceIdx <= numNodes && isfinite(e.Alfa)
                                delta_in(sourceIdx) = e.Alfa / denominator;
                            end
                        end
                        delta_in_cache{i} = delta_in;

                        % Коэффициенты δ_Out для исходящих рёбер
                        delta_out = zeros(1, numNodes);
                        for edge_idx = 1:numel(outgoingEdges)
                            e = outgoingEdges(edge_idx);
                            targetNode = e.TargetNode;
                            targetIdx = find(nodes == targetNode, 1);
                            if ~isempty(targetIdx) && targetIdx <= numNodes && isfinite(e.Alfa)
                                % Для целевой вершины вычисляем её знаменатель
                                targetOutEdges = obj.outgoingEdgesCache{targetIdx};
                                if ~isempty(targetOutEdges)
                                    target_alpha_sum = 0;
                                    for t_edge_idx = 1:numel(targetOutEdges)
                                        te = targetOutEdges(t_edge_idx);
                                        if isfinite(te.Alfa)
                                            target_alpha_sum = target_alpha_sum + te.Alfa;
                                        end
                                    end
                                    target_alpha_sum = target_alpha_sum + 1;
                                    
                                    % Вычисляем Γ для целевой вершины
                                    target_Gamma = 0;
                                    for t_edge_idx = 1:numel(targetOutEdges)
                                        te = targetOutEdges(t_edge_idx);
                                        t_targetNode = te.TargetNode;
                                        t_targetIdx = find(nodes == t_targetNode, 1);
                                        if ~isempty(t_targetIdx) && t_targetIdx <= length(modelValues)
                                            t_F_u = modelValues(t_targetIdx);
                                            if isprop(te, 'Delta') && isfinite(te.Delta) && isfinite(t_F_u)
                                                target_Gamma = target_Gamma + te.Delta * t_F_u;
                                            end
                                        end
                                    end
                                    
                                    target_denominator = target_alpha_sum - target_Gamma;
                                    if abs(target_denominator) < 1e-10
                                        target_denominator = sign(target_denominator) * 1e-10;
                                    end
                                    
                                    delta_out(targetIdx) = e.Alfa / target_denominator;
                                end
                            end
                        end
                        delta_out_cache{i} = delta_out;
                    end

                    % 2. Распространение ошибок от белых вершин к черным
                    blackIndices = obj.blackNodeIndices(obj.blackNodeIndices <= numNodes);
                    
                    % Итеративное распространение
                    maxIter = 20; % Уменьшаем количество итераций для стабильности
                    tolerance = 1e-6;

                    for iter = 1:maxIter
                        J_prev = J_total;

                        % Для каждой черной вершины вычисляем J
                        for i = blackIndices
                            delta_in = delta_in_cache{i};
                            delta_out = delta_out_cache{i};

                            % Сумма по входящим соседям
                            sum_in = 0;
                            incomingNeighbors = obj.incomingNeighborsCache{i};
                            for neighbor = incomingNeighbors
                                neighborIdx = find(nodes == neighbor, 1);
                                if ~isempty(neighborIdx) && neighborIdx <= length(J_total)
                                    if neighborIdx <= length(delta_in) && delta_in(neighborIdx) ~= 0
                                        sum_in = sum_in + delta_in(neighborIdx) * J_total(neighborIdx);
                                    end
                                end
                            end

                            % Сумма по исходящим соседям
                            sum_out = 0;
                            outgoingEdges = obj.outgoingEdgesCache{i};
                            for e = outgoingEdges
                                targetNode = e.TargetNode;
                                targetIdx = find(nodes == targetNode, 1);
                                if ~isempty(targetIdx) && targetIdx <= length(J_total)
                                    if targetIdx <= length(delta_out) && delta_out(targetIdx) ~= 0
                                        sum_out = sum_out + delta_out(targetIdx) * J_total(targetIdx);
                                    end
                                end
                            end

                            % Общая ошибка для черной вершины
                            J_black(i) = sum_in + sum_out;
                            J_total(i) = J_black(i);
                        end

                        % Проверка сходимости
                        if max(abs(J_total - J_prev)) < tolerance
                            break;
                        end
                    end
                end

                % --- Вычисление градиентов для всех 4 параметров ---
                % Очистка кеша производных графа
                obj.graph.ClearDerivativeCache();
                
                for i = 1:numNodes
                    if i > length(J_total) || J_total(i) == 0, continue; end

                    currentNode = nodes(i);
                    outgoingEdges = obj.outgoingEdgesCache{i};
                    incomingEdges = obj.incomingEdgesCache{i};
                    weight = NodesWeights(min(i, length(NodesWeights)));

                    % Градиенты для исходящих рёбер (alpha, beta)
                    for j = 1:numel(outgoingEdges)
                        if j > length(batchAlGrad{i}) || j > length(batchBtGrad{i})
                            continue;
                        end

                        edge = outgoingEdges(j);

                        try
                            % Производные для исходящих ребер
                            dF_dalpha_out = obj.graph.computeAlphaDerivativeRecursive(i, j, 'outgoing');
                            dF_dbeta_out = obj.graph.computeBetaDerivativeRecursive(i, j, 'outgoing');
                            
                            if isfinite(dF_dalpha_out) && isfinite(J_total(i))
                                batchAlGrad{i}(j) = batchAlGrad{i}(j) - dF_dalpha_out * J_total(i) * weight;
                            end
                            
                            if isfinite(dF_dbeta_out) && isfinite(J_total(i))
                                batchBtGrad{i}(j) = batchBtGrad{i}(j) - dF_dbeta_out * J_total(i) * weight;
                            end
                        catch
                            % Пропускаем ошибки в расчете производных
                            continue;
                        end
                    end

                    % Градиенты для входящих рёбер (alpha, beta, gamma, delta)
                    for j = 1:numel(incomingEdges)
                        edge = incomingEdges(j);
                        sourceNode = edge.SourceNode;
                        sourceIdx = find(nodes == sourceNode, 1);

                        if isempty(sourceIdx) || sourceIdx > numNodes, continue; end

                        % Находим позицию этого ребра в исходящих рёбрах sourceNode
                        sourceOutgoing = obj.outgoingEdgesCache{sourceIdx};
                        if isempty(sourceOutgoing), continue; end
                        
                        edgePos = 0;
                        for pos = 1:numel(sourceOutgoing)
                            if sourceOutgoing(pos).TargetNode == currentNode
                                edgePos = pos;
                                break;
                            end
                        end
                        
                        if edgePos == 0 || edgePos > length(sourceOutgoing)
                            continue;
                        end

                        try
                            sourceWeight = NodesWeights(min(sourceIdx, length(NodesWeights)));
                            
                            % Производные для входящих ребер
                            dF_dalpha_in = obj.graph.computeAlphaDerivativeRecursive(i, j, 'incoming');
                            dF_dbeta_in = obj.graph.computeBetaDerivativeRecursive(i, j, 'incoming');
                            dF_dgamma = obj.graph.computeGammaDerivativeRecursive(i, j);
                            dF_ddelta = obj.graph.computeDeltaDerivativeRecursive(i, j);
                            
                            % Обновление градиентов с проверкой на finite
                            if isfinite(dF_dalpha_in) && isfinite(J_total(i))
                                if edgePos <= length(batchAlGrad{sourceIdx})
                                    batchAlGrad{sourceIdx}(edgePos) = batchAlGrad{sourceIdx}(edgePos) - ...
                                        dF_dalpha_in * J_total(i) * sourceWeight;
                                end
                            end
                            
                            if isfinite(dF_dbeta_in) && isfinite(J_total(i))
                                if edgePos <= length(batchBtGrad{sourceIdx})
                                    batchBtGrad{sourceIdx}(edgePos) = batchBtGrad{sourceIdx}(edgePos) - ...
                                        dF_dbeta_in * J_total(i) * sourceWeight;
                                end
                            end
                            
                            if isfinite(dF_dgamma) && isfinite(J_total(i))
                                if edgePos <= length(batchGmGrad{sourceIdx})
                                    batchGmGrad{sourceIdx}(edgePos) = batchGmGrad{sourceIdx}(edgePos) - ...
                                        dF_dgamma * J_total(i) * sourceWeight;
                                end
                            end
                            
                            if isfinite(dF_ddelta) && isfinite(J_total(i))
                                if edgePos <= length(batchDlGrad{sourceIdx})
                                    batchDlGrad{sourceIdx}(edgePos) = batchDlGrad{sourceIdx}(edgePos) - ...
                                        dF_ddelta * J_total(i) * sourceWeight;
                                end
                            end
                        catch
                            % Пропускаем ошибки в расчете производных
                            continue;
                        end
                    end
                end
                
                validSamples = validSamples + 1;
                
            catch ME
                warning('Ошибка при обработке примера %d: %s', sampleIdx, ME.message);
                continue;
            end
        end

        % Если нет валидных примеров в батче, пропускаем обновление
        if validSamples == 0
            continue;
        end

        % --- Нормализация и обрезка градиентов ---
        invNumValidSamples = 1 / validSamples;
        for i = 1:numNodes
            if ~isempty(batchAlGrad{i})
                grad = batchAlGrad{i} * invNumValidSamples;
                grad = max(min(grad, ClipUp), ClipDown);
                grad(~isfinite(grad)) = 0;
                batchAlGrad{i} = grad;
            end
            if ~isempty(batchBtGrad{i})
                grad = batchBtGrad{i} * invNumValidSamples;
                grad = max(min(grad, ClipUp), ClipDown);
                grad(~isfinite(grad)) = 0;
                batchBtGrad{i} = grad;
            end
            if ~isempty(batchGmGrad{i})
                grad = batchGmGrad{i} * invNumValidSamples;
                grad = max(min(grad, ClipUp), ClipDown);
                grad(~isfinite(grad)) = 0;
                batchGmGrad{i} = grad;
            end
            if ~isempty(batchDlGrad{i})
                grad = batchDlGrad{i} * invNumValidSamples;
                grad = max(min(grad, ClipUp), ClipDown);
                grad(~isfinite(grad)) = 0;
                batchDlGrad{i} = grad;
            end
        end

        % --- Добавление L2 регуляризации ---
        for i = 1:numNodes
            edges = obj.outgoingEdgesCache{i};
            for j = 1:numel(edges)
                if j > length(edges), continue; end
                
                edge = edges(j);
                
                % Регуляризация для alpha и beta
                if j <= length(batchAlGrad{i}) && isfinite(edge.Alfa)
                    batchAlGrad{i}(j) = batchAlGrad{i}(j) + Lambda * edge.Alfa;
                end
                if j <= length(batchBtGrad{i}) && isfinite(edge.Beta)
                    batchBtGrad{i}(j) = batchBtGrad{i}(j) + Lambda * edge.Beta;
                end
                
                % Регуляризация для gamma и delta (если они существуют)
                if isprop(edge, 'Gamma') && j <= length(batchGmGrad{i}) && isfinite(edge.Gamma)
                    batchGmGrad{i}(j) = batchGmGrad{i}(j) + Lambda * edge.Gamma;
                end
                if isprop(edge, 'Delta') && j <= length(batchDlGrad{i}) && isfinite(edge.Delta)
                    batchDlGrad{i}(j) = batchDlGrad{i}(j) + Lambda * edge.Delta;
                end
            end
        end

        % --- Обновление параметров с помощью ADAM для всех 4 параметров ---
        beta1_t = beta1^obj.t;
        beta2_t = beta2^obj.t;
        mCorrFactor = 1 / (1 - beta1_t);
        vCorrFactor = 1 / (1 - beta2_t);

        for i = 1:numNodes
            edges = obj.outgoingEdgesCache{i};
            if isempty(edges), continue; end

            % Обновление моментов ADAM для всех параметров
            try
                % Alpha
                if ~isempty(batchAlGrad{i})
                    obj.mAl{i} = beta1 * obj.mAl{i} + (1-beta1) * batchAlGrad{i};
                    obj.vAl{i} = beta2 * obj.vAl{i} + (1-beta2) * (batchAlGrad{i}.^2);
                    
                    % Обновление параметров alpha
                    sqrtVAl = sqrt(obj.vAl{i} * vCorrFactor) + epsilon;
                    alfaUpdates = obj.learningRate * NodesLRScale(min(i, length(NodesLRScale))) * ...
                                  (obj.mAl{i} * mCorrFactor) ./ sqrtVAl;
                    
                    for j = 1:min(length(edges), length(alfaUpdates))
                        if isfinite(edges(j).Alfa) && isfinite(alfaUpdates(j))
                            edges(j).Alfa = edges(j).Alfa - alfaUpdates(j);
                        end
                    end
                end
                
                % Beta
                if ~isempty(batchBtGrad{i})
                    obj.mBt{i} = beta1 * obj.mBt{i} + (1-beta1) * batchBtGrad{i};
                    obj.vBt{i} = beta2 * obj.vBt{i} + (1-beta2) * (batchBtGrad{i}.^2);
                    
                    % Обновление параметров beta
                    sqrtVBt = sqrt(obj.vBt{i} * vCorrFactor) + epsilon;
                    betaUpdates = obj.learningRate * NodesLRScale(min(i, length(NodesLRScale))) * ...
                                  (obj.mBt{i} * mCorrFactor) ./ sqrtVBt;
                    
                    for j = 1:min(length(edges), length(betaUpdates))
                        if isfinite(edges(j).Beta) && isfinite(betaUpdates(j))
                            edges(j).Beta = edges(j).Beta - betaUpdates(j);
                        end
                    end
                end
                
                % Gamma
                if ~isempty(batchGmGrad{i}) && ~all(batchGmGrad{i} == 0)
                    obj.mGm{i} = beta1 * obj.mGm{i} + (1-beta1) * batchGmGrad{i};
                    obj.vGm{i} = beta2 * obj.vGm{i} + (1-beta2) * (batchGmGrad{i}.^2);
                    
                    % Обновление параметров gamma
                    for j = 1:min(length(edges), length(obj.mGm{i}))
                        if isprop(edges(j), 'Gamma') && isfinite(edges(j).Gamma)
                            sqrtVGm = sqrt(obj.vGm{i}(j) * vCorrFactor) + epsilon;
                            gammaUpdate = obj.learningRate * NodesLRScale(min(i, length(NodesLRScale))) * ...
                                         (obj.mGm{i}(j) * mCorrFactor) / sqrtVGm;
                            
                            if isfinite(gammaUpdate)
                                edges(j).Gamma = edges(j).Gamma - gammaUpdate;
                            end
                        end
                    end
                end
                
                % Delta
                if ~isempty(batchDlGrad{i}) && ~all(batchDlGrad{i} == 0)
                    obj.mDl{i} = beta1 * obj.mDl{i} + (1-beta1) * batchDlGrad{i};
                    obj.vDl{i} = beta2 * obj.vDl{i} + (1-beta2) * (batchDlGrad{i}.^2);
                    
                    % Обновление параметров delta
                    for j = 1:min(length(edges), length(obj.mDl{i}))
                        if isprop(edges(j), 'Delta') && isfinite(edges(j).Delta)
                            sqrtVDl = sqrt(obj.vDl{i}(j) * vCorrFactor) + epsilon;
                            deltaUpdate = obj.learningRate * NodesLRScale(min(i, length(NodesLRScale))) * ...
                                         (obj.mDl{i}(j) * mCorrFactor) / sqrtVDl;
                            
                            if isfinite(deltaUpdate)
                                edges(j).Delta = edges(j).Delta - deltaUpdate;
                            end
                        end
                    end
                end
                
            catch ME
                warning('Ошибка при обновлении параметров для вершины %d: %s', i, ME.message);
                continue;
            end
        end
    end
end
       
    end
end