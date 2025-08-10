classdef Trainer < handle

    properties (Access = private)
        % Кешированные данные о графе
        whiteNodeIndices
        blackNodeIndices
        incomingEdgesCache
        outgoingEdgesCache
        incomingNeighborsCache
        distancesCache

        % Моменты ADAM
        mAl
        vAl
        mBt
        vBt
        t
        % Остальные параметры
        epochsNoImprove = 0;      % Фактическое количество эпох без улучшения
        bestAl                     % Лучшие альфа-значения по результатам настройки
        bestBt                     % Лучшие бета-значения по результатам настройки
    end

    properties
        graph BWGraph.GraphShell    % Графовая модель
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
        function obj = Trainer(graph, BatchSize)
            arguments
                graph       BWGraph.GraphShell
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
                obj             BWGraph.Trainer.Trainer
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
            numWhiteNodes = obj.graph.GetNumOfWhiteNode;
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
            % Создаем 4 субграфика в 2 колонки
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
                obj.Compute_V5(XDataTrain, YDataTrain, ...
                    Beta1, Beta2, Eps, NodeWeight, NodeSize, ClipUp, ClipDown, Lambda, Lambda_Agg, targetNodeIndices);

                % Расчет ошибки на обучающей выборке
                trainEerror = obj.CalculateError(XDataTrain, YDataTrain, targetNodeIndices, errorMetric);
                obj.trainErrors(end+1) = trainEerror;

                % Расчет ошибки на тестовой выборке
                testError = obj.CalculateError(XDataTest, YDataTest, targetNodeIndices, errorMetric);
                obj.testErrors(end+1) = testError;

                % Вычисление разницы между ошибками
                errorDiffs(epoch) = testError - trainEerror;

                % Проверка увеличения разницы ошибок (переобучение)
                if epoch > 1
                    if errorDiffs(epoch) > errorDiffs(epoch-1) + 1e-3
                        diffIncreaseCount = diffIncreaseCount + 1;
                        fprintf('Увеличение разницы ошибок (переобучение) %d/10\n', diffIncreaseCount);
                    else
                        diffIncreaseCount = max(0, diffIncreaseCount - 1);
                    end
                end

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

                % 1. Критерий переобучения (увеличение разницы ошибок)
                if diffIncreaseCount >= 10
                    stopTraining = true;
                    stopReason = 'Обнаружено переобучение (разница ошибок увеличивается 5 эпох подряд)';
                end

                % 2. Критерий плато
                if obj.epochsNoImprove >= obj.patience
                    obj.plateauCount = obj.plateauCount + 1;
                    fprintf('Обнаружено плато ошибки (попытка %d из %d) на эпохе %d\n', ...
                        obj.plateauCount, obj.maxPlateauCount, epoch);

                    if obj.plateauCount >= obj.maxPlateauCount
                        stopTraining = true;
                        stopReason = 'Достигнуто максимальное количество плато';
                    else
                        % Случайное смещение параметров
                        obj.RandomShiftParameters();
                        % Запоминаем эпоху смещения
                        shiftEpochs(end+1) = epoch;
                        % Сброс счетчиков и learning rate
                        obj.epochsNoImprove = 0;
                        diffIncreaseCount = 0;
                        obj.learningRate = LearningRate;
                        fprintf('Применено случайное смещение параметров. Продолжение обучения...\n');
                    end
                end

                % 3. Критерий достижения целевой ошибки
                if obj.bestTestError < TargetError
                    stopTraining = true;
                    stopReason = sprintf('Достигнута целевая ошибка (MAE < %3f)', TargetError);
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
                        epoch, trainEerror, testError, obj.learningRate, epochTimes(epoch));
                else
                    % Вывод прогресса
                    fprintf('Эпоха %3d: Train MAPE = %.4f | Test MAPE = %.4f | LR = %.2e | Время = %.2f сек\n',...
                        epoch, trainEerror, testError, obj.learningRate, epochTimes(epoch));
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

            % Создаем матрицы для хранения всех alfa и beta
            obj.bestAl = zeros(numNodes, numNodes);
            obj.bestBt = zeros(numNodes, numNodes);

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

        function SaveBestParameters(obj)
            % Сохраняет текущие параметры графа как лучшие
            nodes = obj.graph.ListOfNodes;
            numNodes = numel(nodes);

            % Очищаем предыдущие лучшие значения
            obj.bestAl = zeros(numNodes, numNodes);
            obj.bestBt = zeros(numNodes, numNodes);

            % Сохраняем текущие значения
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
                end
            end
        end

        function RandomShiftParameters(obj)
            % Применяет случайное смещение к параметрам графа
            nodes = obj.graph.ListOfNodes;
            for i = 1:numel(nodes)
                node = nodes(i);
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
            end

            % 4. Финальное усреднение
            switch lower(errorMetric)
                case 'mae'
                    errorValue = totalError / totalPoints;
                case 'mape'
                    errorValue = (totalError / totalPoints) * 100; % В процентах
            end
        end

        function Compute_V3(obj, XData, YData, Beta1, Beta2, Eps, NodesWeights, NodesLRScale, ClipUp, ClipDown, Lambda)
            arguments
                obj             BWGraph.Trainer.Trainer,
                XData
                YData
                Beta1           {mustBePositive, mustBeFinite},
                Beta2           {mustBePositive, mustBeFinite},
                Eps             {mustBePositive, mustBeFinite},
                NodesWeights    (1,:)      % Веса вершин для степени настройки параметров
                NodesLRScale    (1,:)      % Масштабирование LR для вершин,
                ClipUp          {mustBeFinite},
                ClipDown        {mustBeFinite},
                Lambda          {mustBeFinite}
            end

            % --- Инициализация параметров ---
            beta1 = Beta1;
            beta2 = Beta2;
            epsilon = Eps;
            nodes = obj.graph.ListOfNodes;
            numNodes = numel(nodes);

            % Инициализация кешей в классе, если они еще не созданы
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

            % --- Инициализация моментов ADAM ---
            if isempty(obj.mAl)
                obj.mAl = cell(numNodes, 1);
                obj.vAl = cell(numNodes, 1);
                obj.mBt = cell(numNodes, 1);
                obj.vBt = cell(numNodes, 1);
                numEdgesPerNode = cellfun(@numel, obj.outgoingEdgesCache);
                for i = 1:numNodes
                    obj.mAl{i} = zeros(1, numEdgesPerNode(i));
                    obj.vAl{i} = zeros(1, numEdgesPerNode(i));
                    obj.mBt{i} = zeros(1, numEdgesPerNode(i));
                    obj.vBt{i} = zeros(1, numEdgesPerNode(i));
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

                % Инициализация градиентов для всего батча
                batchAlGrad = cell(numNodes, 1);
                batchBtGrad = cell(numNodes, 1);
                for i = 1:numNodes
                    numEdges = numel(obj.outgoingEdgesCache{i});
                    batchAlGrad{i} = zeros(1, numEdges);
                    batchBtGrad{i} = zeros(1, numEdges);
                end

                % --- Обработка примеров в батче ---
                for k = 1:numInBatch
                    sampleIdx = batchIndices(k);
                    xMatrix = XData(sampleIdx);
                    yMatrix = YData(sampleIdx);

                    % Прямой проход
                    obj.graph.Forward(xMatrix);
                    modelValues = obj.graph.GetModelResults();
                    targetValues = yMatrix.getRow(1);

                    % Вычисление ошибок
                    errors = zeros(1, numNodes);
                    errors(obj.whiteNodeIndices) = modelValues(obj.whiteNodeIndices) - targetValues;

                    % Распространение ошибок для черных вершин
                    if ~isempty(obj.blackNodeIndices)
                        % Сначала обрабатываем белые вершины (источники ошибок)
                        whiteErrors = errors(obj.whiteNodeIndices);
                        whiteNodes = nodes(obj.whiteNodeIndices);

                        % Затем распространяем ошибки на черные вершины
                        % Используем итеративный подход, так как граф может быть цикличным
                        maxIter = 10; % Максимальное число итераций для сходимости
                        for iter = 1:maxIter
                            prevErrors = errors;

                            if ~any(errors == 0)
                                break;
                            end
                            % Обход всех черных вершин
                            for i = obj.blackNodeIndices
                                incomingEdges = obj.incomingEdgesCache{i};
                                outgoingEdges = obj.outgoingEdgesCache{i};

                                errorFromIncoming = 0;
                                errorFromOutgoing = 0;
                                if ~isempty(incomingEdges)
                                    % Суммируем взвешенные ошибки от входящих соседей
                                    for e = incomingEdges
                                        u = e.SourceNode;
                                        uIdx = find(nodes == u, 1);
                                        alpha_e = e.Alfa;
                                        errorFromIncoming = errorFromIncoming + alpha_e * errors(uIdx);
                                    end
                                end


                                % Суммируем взвешенные ошибки от исходящих соседей
                                if ~isempty(outgoingEdges)
                                    for e = outgoingEdges
                                        v = e.TargetNode;
                                        vIdx = find(nodes == v, 1);
                                        sumAlphaOutV = sum(arrayfun(@(x) x.Alfa, obj.outgoingEdgesCache{vIdx}));
                                        alpha_e = e.Alfa;
                                        errorFromOutgoing = errorFromOutgoing + (alpha_e * errors(vIdx)) / (sumAlphaOutV + 1) ;
                                    end
                                end

                                % Итоговая ошибка для черной вершины
                                sumAlphaOut = sum(arrayfun(@(e) e.Alfa, outgoingEdges));
                                errors(i) = (errorFromIncoming + errorFromOutgoing) / (sumAlphaOut + 1);
                            end

                            % Проверка сходимости
                            if max(abs(errors - prevErrors)) < 1e-6
                                break;
                            end
                        end
                    end

                    % Предварительно вычисляем производные для всех узлов
                    alphaDerivativesCache = cell(numNodes, 1);
                    betaDerivativesCache = cell(numNodes, 1);
                    for i = 1:numNodes
                        if errors(i) == 0, continue; end
                        alphaDerivativesCache{i} = obj.graph.computeAlphaDerivatives(i);
                        betaDerivativesCache{i} = obj.graph.computeBetaDerivatives(i);
                    end

                    % Вычисление градиентов для каждого узла
                    for i = 1:numNodes
                        if errors(i) == 0, continue; end

                        edges = obj.outgoingEdgesCache{i};
                        alphaDerivs = alphaDerivativesCache{i};
                        betaDerivs = betaDerivativesCache{i};
                        weight = NodesWeights(i);

                        % Обработка исходящих ребер (alpha)
                        if ~isempty(alphaDerivs.outgoing)
                            edgeIndices = [alphaDerivs.outgoing.edge];
                            [~, edgeOrder] = ismember(edgeIndices, edges);
                            derivatives = [alphaDerivs.outgoing.derivative];
                            batchAlGrad{i}(edgeOrder) = batchAlGrad{i}(edgeOrder) - ...
                                derivatives * errors(i) * weight;
                        end

                        % Обработка исходящих ребер (beta)
                        if ~isempty(betaDerivs.outgoing)
                            edgeIndices = [betaDerivs.outgoing.edge];
                            [~, edgeOrder] = ismember(edgeIndices, edges);
                            derivatives = [betaDerivs.outgoing.derivative];
                            batchBtGrad{i}(edgeOrder) = batchBtGrad{i}(edgeOrder) - ...
                                derivatives * errors(i) * weight;
                        end
                    end
                end

                % --- Нормализация и обрезка градиентов ---
                invNumInBatch = 1 / numInBatch;
                for i = 1:numNodes
                    if ~isempty(batchAlGrad{i})
                        batchAlGrad{i} = min(max(batchAlGrad{i} * invNumInBatch, ClipDown), ClipUp);
                    end
                    if ~isempty(batchBtGrad{i})
                        batchBtGrad{i} = min(max(batchBtGrad{i} * invNumInBatch, ClipDown), ClipUp);
                    end
                end

                % --- Оптимизированное обновление параметров с помощью ADAM ---
                beta1_t = beta1^obj.t;
                beta2_t = beta2^obj.t;
                mCorrFactor = 1 / (1 - beta1_t);
                vCorrFactor = 1 / (1 - beta2_t);

                for i = 1:numNodes
                    edges = obj.outgoingEdgesCache{i};
                    if isempty(edges), continue; end

                    % Обновление моментов для alpha
                    obj.mAl{i} = beta1 * obj.mAl{i} + (1-beta1) * batchAlGrad{i};
                    obj.vAl{i} = beta2 * obj.vAl{i} + (1-beta2) * (batchAlGrad{i}.^2);

                    % Обновление моментов для beta
                    obj.mBt{i} = beta1 * obj.mBt{i} + (1-beta1) * batchBtGrad{i};
                    obj.vBt{i} = beta2 * obj.vBt{i} + (1-beta2) * (batchBtGrad{i}.^2);

                    % Применение обновлений
                    lr = obj.learningRate * NodesLRScale(i);
                    sqrtVAl = sqrt(obj.vAl{i} * vCorrFactor) + epsilon;
                    sqrtVBt = sqrt(obj.vBt{i} * vCorrFactor) + epsilon;

                    alfaUpdates = lr * (obj.mAl{i} * mCorrFactor) ./ sqrtVAl;
                    betaUpdates = lr * (obj.mBt{i} * mCorrFactor) ./ sqrtVBt;

                    % Применяем обновления к ребрам по одному
                    for j = 1:numel(edges)
                        edges(j).Alfa = edges(j).Alfa + alfaUpdates(j);
                        edges(j).Beta = edges(j).Beta + betaUpdates(j);
                    end
                end
            end
        end

        function Compute_V4(obj, XData, YData, Beta1, Beta2, Eps, NodesWeights, NodesLRScale, ClipUp, ClipDown, Lambda, lambda_Agg, targetWhiteIndices)
            arguments
                obj                 BWGraph.Trainer.Trainer,
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

            % Инициализация моментов ADAM
            if isempty(obj.mAl)
                obj.mAl = cell(numNodes, 1);
                obj.vAl = cell(numNodes, 1);
                obj.mBt = cell(numNodes, 1);
                obj.vBt = cell(numNodes, 1);
                numEdgesPerNode = cellfun(@numel, obj.outgoingEdgesCache);
                for i = 1:numNodes
                    obj.mAl{i} = zeros(1, numEdgesPerNode(i));
                    obj.vAl{i} = zeros(1, numEdgesPerNode(i));
                    obj.mBt{i} = zeros(1, numEdgesPerNode(i));
                    obj.vBt{i} = zeros(1, numEdgesPerNode(i));
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

                % Инициализация градиентов для всего батча
                batchAlGrad = cell(numNodes, 1);
                batchBtGrad = cell(numNodes, 1);
                for i = 1:numNodes
                    numEdges = numel(obj.outgoingEdgesCache{i});
                    batchAlGrad{i} = zeros(1, numEdges);
                    batchBtGrad{i} = zeros(1, numEdges);
                end

                % --- Обработка примеров в батче ---
                for k = 1:numInBatch
                    sampleIdx = batchIndices(k);
                    xMatrix = XData(sampleIdx);
                    yMatrix = YData(sampleIdx);

                    % Прямой проход
                    obj.graph.Forward(xMatrix);
                    modelValues = obj.graph.GetModelResults();
                    targetValues = yMatrix.getRow(1);

                    % Вычисление ошибок для белых вершин
                    errors = zeros(1, numNodes);
                    errors(obj.whiteNodeIndices) = modelValues(obj.whiteNodeIndices) - targetValues;

                    % Расчет усредненной ошибки для целевых белых вершин
                    targetErrors = errors(targetWhiteIndices);
                    meanTargetError = mean(targetErrors);

                    % Добавление усредненной ошибки к белым вершинам
                    errors(obj.whiteNodeIndices) = errors(obj.whiteNodeIndices) + lambda_Agg * meanTargetError;

                    % Распространение ошибок для черных вершин
                    if ~isempty(obj.blackNodeIndices)
                        maxIter = 10;
                        for iter = 1:maxIter
                            prevErrors = errors;

                            for i = obj.blackNodeIndices
                                incomingEdges = obj.incomingEdgesCache{i};
                                outgoingEdges = obj.outgoingEdgesCache{i};

                                errorFromIncoming = 0;
                                errorFromOutgoing = 0;

                                % Взвешенная сумма от входящих соседей
                                if ~isempty(incomingEdges)
                                    for e = incomingEdges
                                        u = e.SourceNode;
                                        uIdx = find(nodes == u, 1);
                                        alpha_e = e.Alfa;
                                        errorFromIncoming = errorFromIncoming + alpha_e * errors(uIdx);
                                    end
                                end

                                % Взвешенная сумма от исходящих соседей
                                if ~isempty(outgoingEdges)
                                    for e = outgoingEdges
                                        v = e.TargetNode;
                                        vIdx = find(nodes == v, 1);
                                        sumAlphaOutV = sum(arrayfun(@(x) x.Alfa, obj.outgoingEdgesCache{vIdx}));
                                        alpha_e = e.Alfa;
                                        errorFromOutgoing = errorFromOutgoing + (alpha_e * errors(vIdx)) / (sumAlphaOutV + 1);
                                    end
                                end

                                % Итоговая ошибка для черной вершины
                                sumAlphaOut = sum(arrayfun(@(e) e.Alfa, outgoingEdges));
                                errors(i) = (errorFromIncoming + errorFromOutgoing) / (sumAlphaOut + 1);
                            end

                            % Проверка сходимости
                            if max(abs(errors - prevErrors)) < 1e-6
                                break;
                            end
                        end
                    end



                    % Предварительное вычисление производных
                    alphaDerivativesCache = cell(numNodes, 1);
                    betaDerivativesCache = cell(numNodes, 1);
                    for i = 1:numNodes
                        if errors(i) == 0, continue; end
                        alphaDerivativesCache{i} = obj.graph.computeAlphaDerivatives(i);
                        betaDerivativesCache{i} = obj.graph.computeBetaDerivatives(i);
                    end

                    % Вычисление градиентов
                    for i = 1:numNodes
                        if errors(i) == 0, continue; end

                        edges = obj.outgoingEdgesCache{i};
                        alphaDerivs = alphaDerivativesCache{i};
                        betaDerivs = betaDerivativesCache{i};
                        weight = NodesWeights(i);

                        % Градиенты для alpha
                        if ~isempty(alphaDerivs.outgoing)
                            edgeIndices = [alphaDerivs.outgoing.edge];
                            [~, edgeOrder] = ismember(edgeIndices, edges);
                            derivatives = [alphaDerivs.outgoing.derivative];

                            % Получаем текущие значения alpha для регуляризации
                            currentAlphas = (arrayfun(@(e) e.Alfa, edges))';

                            batchAlGrad{i}(edgeOrder) = batchAlGrad{i}(edgeOrder) - ...
                                derivatives * errors(i) * weight + Lambda * currentAlphas;
                        end

                        % Градиенты для beta
                        if ~isempty(betaDerivs.outgoing)
                            edgeIndices = [betaDerivs.outgoing.edge];
                            [~, edgeOrder] = ismember(edgeIndices, edges);
                            derivatives = [betaDerivs.outgoing.derivative];

                            currentBetas = (arrayfun(@(e) e.Beta, edges))';

                            regul = derivatives * errors(i) * weight + currentBetas * Lambda;

                            batchBtGrad{i}(edgeOrder) = batchBtGrad{i}(edgeOrder) - regul;
                        end
                    end
                end

                % --- Нормализация и обрезка градиентов ---
                invNumInBatch = 1 / numInBatch;
                for i = 1:numNodes
                    if ~isempty(batchAlGrad{i})
                        batchAlGrad{i} = min(max(batchAlGrad{i} * invNumInBatch, ClipDown), ClipUp);
                    end
                    if ~isempty(batchBtGrad{i})
                        batchBtGrad{i} = min(max(batchBtGrad{i} * invNumInBatch, ClipDown), ClipUp);
                    end
                end

                % --- Обновление параметров с помощью ADAM ---
                beta1_t = beta1^obj.t;
                beta2_t = beta2^obj.t;
                mCorrFactor = 1 / (1 - beta1_t);
                vCorrFactor = 1 / (1 - beta2_t);

                for i = 1:numNodes
                    edges = obj.outgoingEdgesCache{i};
                    if isempty(edges), continue; end

                    % Обновление моментов
                    obj.mAl{i} = beta1 * obj.mAl{i} + (1-beta1) * batchAlGrad{i};
                    obj.vAl{i} = beta2 * obj.vAl{i} + (1-beta2) * (batchAlGrad{i}.^2);
                    obj.mBt{i} = beta1 * obj.mBt{i} + (1-beta1) * batchBtGrad{i};
                    obj.vBt{i} = beta2 * obj.vBt{i} + (1-beta2) * (batchBtGrad{i}.^2);

                    % Применение обновлений
                    lr = obj.learningRate * NodesLRScale(i);
                    sqrtVAl = sqrt(obj.vAl{i} * vCorrFactor) + epsilon;
                    sqrtVBt = sqrt(obj.vBt{i} * vCorrFactor) + epsilon;

                    alfaUpdates = lr * (obj.mAl{i} * mCorrFactor) ./ sqrtVAl;
                    betaUpdates = lr * (obj.mBt{i} * mCorrFactor) ./ sqrtVBt;

                    for j = 1:numel(edges)
                        edges(j).Alfa = edges(j).Alfa + alfaUpdates(j);
                        edges(j).Beta = edges(j).Beta + betaUpdates(j);
                    end
                end
            end
        end

        function Compute_V5(obj, XData, YData, Beta1, Beta2, Eps, NodesWeights, NodesLRScale, ClipUp, ClipDown, Lambda, lambda_Agg, targetWhiteIndices)
            arguments
                obj                 BWGraph.Trainer.Trainer,
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

            % Инициализация моментов ADAM
            if isempty(obj.mAl)
                obj.mAl = cell(numNodes, 1);
                obj.vAl = cell(numNodes, 1);
                obj.mBt = cell(numNodes, 1);
                obj.vBt = cell(numNodes, 1);
                numEdgesPerNode = cellfun(@numel, obj.outgoingEdgesCache);
                for i = 1:numNodes
                    obj.mAl{i} = zeros(1, numEdgesPerNode(i));
                    obj.vAl{i} = zeros(1, numEdgesPerNode(i));
                    obj.mBt{i} = zeros(1, numEdgesPerNode(i));
                    obj.vBt{i} = zeros(1, numEdgesPerNode(i));
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

                % Инициализация градиентов для всего батча
                batchAlGrad = cell(numNodes, 1);
                batchBtGrad = cell(numNodes, 1);
                for i = 1:numNodes
                    numEdges = numel(obj.outgoingEdgesCache{i});
                    batchAlGrad{i} = zeros(1, numEdges);
                    batchBtGrad{i} = zeros(1, numEdges);
                end

                % --- Обработка примеров в батче ---
                for k = 1:numInBatch
                    sampleIdx = batchIndices(k);
                    xMatrix = XData(sampleIdx);
                    yMatrix = YData(sampleIdx);

                    % Прямой проход
                    obj.graph.Forward(xMatrix);
                    modelValues = obj.graph.GetModelResults();
                    targetValues = yMatrix.getRow(1);

                    % Вычисление ошибок для белых вершин
                    errors = zeros(1, numNodes);
                    errors(obj.whiteNodeIndices) = modelValues(obj.whiteNodeIndices) - targetValues;

                    % Расчет усредненной ошибки для целевых белых вершин
                    targetErrors = errors(targetWhiteIndices);
                    meanTargetError = mean(targetErrors);

                    % Добавление усредненной ошибки к белым вершинам
                    errors(obj.whiteNodeIndices) = errors(obj.whiteNodeIndices) + lambda_Agg * meanTargetError;

                    % Распространение ошибок для всех вершин (включая черные)
                    maxIter = 10;
                    for iter = 1:maxIter
                        prevErrors = errors;

                        % Обновление ошибок для всех вершин
                        for i = 1:numNodes
                            incomingEdges = obj.incomingEdgesCache{i};
                            outgoingEdges = obj.outgoingEdgesCache{i};

                            % Пропускаем белые вершины (их ошибки уже вычислены)
                            if ismember(i, obj.whiteNodeIndices)
                                continue;
                            end

                            errorFromIncoming = 0;
                            errorFromOutgoing = 0;

                            % Взвешенная сумма от входящих соседей
                            if ~isempty(incomingEdges)
                                for e = incomingEdges
                                    u = e.SourceNode;
                                    uIdx = find(nodes == u, 1);
                                    alpha_e = e.Alfa;
                                    errorFromIncoming = errorFromIncoming + alpha_e * errors(uIdx);
                                end
                            end

                            % Взвешенная сумма от исходящих соседей
                            if ~isempty(outgoingEdges)
                                for e = outgoingEdges
                                    v = e.TargetNode;
                                    vIdx = find(nodes == v, 1);
                                    sumAlphaOutV = sum(arrayfun(@(x) x.Alfa, obj.outgoingEdgesCache{vIdx}));
                                    alpha_e = e.Alfa;
                                    errorFromOutgoing = errorFromOutgoing + (alpha_e * errors(vIdx)) / (sumAlphaOutV + 1);
                                end
                            end

                            % Итоговая ошибка для вершины
                            sumAlphaOut = sum(arrayfun(@(e) e.Alfa, outgoingEdges));
                            errors(i) = (errorFromIncoming + errorFromOutgoing) / (sumAlphaOut + 1);
                        end

                        % Проверка сходимости
                        if max(abs(errors - prevErrors)) < 1e-6
                            break;
                        end
                    end

                    % % Предварительное вычисление производных для всех вершин
                    % alphaDerivativesCache = cell(numNodes, 1);
                    % betaDerivativesCache = cell(numNodes, 1);
                    % for i = 1:numNodes
                    %     if errors(i) == 0, continue; end
                    %     alphaDerivativesCache{i} = obj.graph.computeAlphaDerivatives(i);
                    %     betaDerivativesCache{i} = obj.graph.computeBetaDerivatives(i);
                    % end

                    % Вычисление градиентов с учетом входящих и исходящих ребер
                    for i = 1:numNodes
                        if errors(i) == 0, continue; end

                        currentNode = nodes(i);
                        edges = obj.outgoingEdgesCache{i};
                        weight = NodesWeights(i);

                        % Градиенты для исходящих ребер (alpha и beta)
                        for j = 1:numel(edges)
                            edge = edges(j);

                            % Производные для исходящих ребер
                            dF_dalpha_out = obj.graph.computeOutgoingAlphaDerivative(i, edge);
                            dF_dbeta_out = obj.graph.computeOutgoingBetaDerivative(i, edge);

                            % Обновление градиентов с регуляризацией
                            batchAlGrad{i}(j) = batchAlGrad{i}(j) - dF_dalpha_out * errors(i) * weight + Lambda * edge.Alfa;
                            batchBtGrad{i}(j) = batchBtGrad{i}(j) - dF_dbeta_out * errors(i) * weight + Lambda * edge.Beta;
                        end

                        % Градиенты для входящих ребер (alpha и beta)
                        incomingEdges = obj.incomingEdgesCache{i};
                        for j = 1:numel(incomingEdges)
                            edge = incomingEdges(j);
                            sourceNode = edge.SourceNode;
                            sourceIdx = find(nodes == sourceNode, 1);

                            if isempty(sourceIdx), continue; end

                            % Производные для входящих ребер
                            dF_dalpha_in = obj.graph.computeIncomingAlphaDerivative(i, edge);
                            dF_dbeta_in = obj.graph.computeIncomingBetaDerivative(i, edge);

                            if ~isnan(dF_dalpha_in)
                                % Находим позицию этого ребра в исходящих ребрах sourceNode
                                sourceEdges = obj.outgoingEdgesCache{sourceIdx};
                                edgePos = find(sourceEdges == edge, 1);

                                if ~isempty(edgePos)
                                    % Обновление градиентов с регуляризацией
                                    batchAlGrad{sourceIdx}(edgePos) = batchAlGrad{sourceIdx}(edgePos) - ...
                                        dF_dalpha_in * errors(i) * NodesWeights(sourceIdx) + Lambda * edge.Alfa;

                                    batchBtGrad{sourceIdx}(edgePos) = batchBtGrad{sourceIdx}(edgePos) - ...
                                        dF_dbeta_in * errors(i) * NodesWeights(sourceIdx) + Lambda * edge.Beta;
                                end
                            end
                        end
                    end
                end

                % --- Нормализация и обрезка градиентов ---
                invNumInBatch = 1 / numInBatch;
                for i = 1:numNodes
                    if ~isempty(batchAlGrad{i})
                        batchAlGrad{i} = min(max(batchAlGrad{i} * invNumInBatch, ClipDown), ClipUp);
                    end
                    if ~isempty(batchBtGrad{i})
                        batchBtGrad{i} = min(max(batchBtGrad{i} * invNumInBatch, ClipDown), ClipUp);
                    end
                end

                % --- Обновление параметров с помощью ADAM ---
                beta1_t = beta1^obj.t;
                beta2_t = beta2^obj.t;
                mCorrFactor = 1 / (1 - beta1_t);
                vCorrFactor = 1 / (1 - beta2_t);

                for i = 1:numNodes
                    edges = obj.outgoingEdgesCache{i};
                    if isempty(edges), continue; end

                    % Обновление моментов
                    obj.mAl{i} = beta1 * obj.mAl{i} + (1-beta1) * batchAlGrad{i};
                    obj.vAl{i} = beta2 * obj.vAl{i} + (1-beta2) * (batchAlGrad{i}.^2);
                    obj.mBt{i} = beta1 * obj.mBt{i} + (1-beta1) * batchBtGrad{i};
                    obj.vBt{i} = beta2 * obj.vBt{i} + (1-beta2) * (batchBtGrad{i}.^2);

                    % Применение обновлений
                    lr = obj.learningRate * NodesLRScale(i);
                    sqrtVAl = sqrt(obj.vAl{i} * vCorrFactor) + epsilon;
                    sqrtVBt = sqrt(obj.vBt{i} * vCorrFactor) + epsilon;

                    alfaUpdates = lr * (obj.mAl{i} * mCorrFactor) ./ sqrtVAl;
                    betaUpdates = lr * (obj.mBt{i} * mCorrFactor) ./ sqrtVBt;

                    for j = 1:numel(edges)
                        edges(j).Alfa = edges(j).Alfa + alfaUpdates(j);
                        edges(j).Beta = edges(j).Beta + betaUpdates(j);
                    end
                end
            end
        end

    end
end

