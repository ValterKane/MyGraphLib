classdef Trainer < handle

    properties (Access = private)
        % Кешированные данные о графе
        whiteNodeIndices
        blackNodeIndices
        incomingEdgesCache
        outgoingEdgesCache
        incomingNeighborsCache
        distancesCache
        edgeSources ;
        edgeTargets;
        edgeAlphas;
        allEdges;

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

        % -----------------------------------
        % Автоматическая калибровка клиппинга
        clipAutoCalibrated (1,1) logical = false
        calibratedClipUpAl   (1,1) double = 0
        calibratedClipDownAl (1,1) double = 0
        calibratedClipUpBt   (1,1) double = 0
        calibratedClipDownBt (1,1) double = 0
        calibratedClipUpGm   (1,1) double = 0
        calibratedClipDownGm (1,1) double = 0
        % Структурный поиск: визуализация
        rejectedEdges           % Матрица [N×2] отклонённых рёбер (src, dst) за текущий шаг
        % Глобальный кеш проверенных рёбер (между шагами поиска)
        globalEdgeCache              % containers.Map: ключ "src->dst" → struct(error, graphHash, removedInStep)
        structuralSearchStepCount = 0  % Счётчик шагов структурного поиска
        structuralSearchConverged = false  % Все комбинации проверены, оптимум найден
        cleanupDone = false                 % Зачистка выполнена (однократно)
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

            % Применяем гиперпараметры α-генерации и перегенерируем α под топологию
            obj.graph.MinAlpha = TrainingOptions.AlphaMin;
            obj.graph.SafetyFactor = TrainingOptions.AlphaSafetyFactor;
            obj.graph.GenerateTopologyAwareAlpha();

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
            figure('Name', 'Training Progress', 'NumberTitle', 'off', 'Position', [100 100 1200 1000]);
            ax1 = subplot(3,3,[1,2]);  % Ошибки (широкий)
            ax2 = subplot(3,3,3);      % LR
            ax3 = subplot(3,3,[4,5]);  % Время эпохи (широкий)
            ax4 = subplot(3,3,6);      % Разница ошибок
            ax5 = subplot(3,3,7);      % Матрица смежности
            ax6 = subplot(3,3,[8,9]);  % Визуальный граф
            obj.rejectedEdges = zeros(0, 2);
            obj.UpdateStructuralPlot(ax5);
            obj.graph.DrawGraph_New([], ax6);

            fprintf('Старт процесса настройки. ЦФ=%s, Метрика=%s\n', obj.TrainingOptions.LossFunction, obj.TrainingOptions.ErrorMetric);

            % Инициализация Алгоритма 1 (рукопись, стр. 488)
            c_no = 0;      % Счётчик эпох без улучшения
            c_p = 0;       % Счётчик выходов из плато (Rp-операций)
            p_e = obj.maxPlateauCount;    % Порог идентификации плато
            p_p = 3;                       % Макс. число Rp-операций
            eta_init = LearningRate;
            topologyChangedEpochs = [];

            for epoch = 1:obj.TrainingOptions.Epoches
                % LR-шедулинг по формуле (3.13): каждые LRDecayInterval эпох
                if mod(epoch, obj.TrainingOptions.LRDecayInterval) == 0
                    LearningRate = max(obj.minLr, eta_init / sqrt(epoch));
                    obj.TrainingOptions.LearningRate = LearningRate;
                end

                epochStart = tic;

                % Основной алгоритм настройки
                obj.Compute_V5(XDataTrain, YDataTrain);

                fprintf('\nНастройка на эпохе No%d завершена!\n',epoch)
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
                    c_no = 0;   % Сброс счётчика без улучшений (Алгоритм 1, стр. 506)
                    obj.SaveBestParameters();
                else
                    c_no = c_no + 1;  % (Алгоритм 1, стр. 509)
                end

                % При застревании — корректировка LR (Алгоритм 1, стр. 511-514)
                if obj.TrainingOptions.EnablePlateauEscape && c_no > 0 && mod(c_no, 5) == 0
                    LearningRate = max(obj.minLr, eta_init / sqrt(epoch));
                    obj.TrainingOptions.LearningRate = LearningRate;
                end

                % Выход из плато через Rp (Алгоритм 1, стр. 515-524)
                if obj.TrainingOptions.EnablePlateauEscape && c_no >= p_e
                    if c_p >= p_p
                        stopTraining = true;
                        stopReason = sprintf('Исчерпаны попытки выхода из плато (c_p=%d)', c_p);
                    else
                        obj.RandomShiftParameters();
                        c_no = 0;
                        c_p = c_p + 1;
                        LearningRate = eta_init;
                        obj.TrainingOptions.LearningRate = LearningRate;
                        fprintf('[Plateau] Rp-оператор применён (c_p=%d/%d)\n', c_p, p_p);
                    end
                end

                % Обнаружение переобучения
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

                % Пунктирные линии в моменты изменения структуры
                yLimits = ylim(ax1);
                for e = topologyChangedEpochs
                    plot(ax1, [e e], yLimits, 'k--', 'LineWidth', 0.7, 'HandleVisibility', 'off');
                end
                hold(ax1, 'off');
                title(ax1, 'Ошибки настройки и тестирования');
                xlabel(ax1, 'Итерация');
                ylabel(ax1, sprintf('%s',obj.TrainingOptions.ErrorMetric));
                legend(ax1, {'Ошибка настройки', 'Ошибка тестирования'}, 'Location', 'best');
                grid(ax1, 'on');

                % График learning rate в логарифмической шкале
                semilogy(ax2, 1:epoch, LearningRate * ones(1, epoch), 'r.', 'MarkerSize', 10);
                hold(ax2, 'on');
                semilogy(ax2, 1:epoch, LearningRate * ones(1, epoch), 'r-', 'LineWidth', 0.5);
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
                title(ax4, 'Разница между ошибками L_{test} и L_{train}');
                xlabel(ax4, 'Итерация');
                ylabel(ax4, '\Delta E');
                legend(ax4, {'Разница ошибок', 'Нулевая линия'}, 'Location', 'best');
                grid(ax4, 'on');

                % Обновление визуального графа (текущие α, β, γ после обучения)
                obj.graph.DrawGraph_New([], ax6);

                drawnow; % Обновляем графики

                % Продолжаем обучение из текущей точки (без отката к лучшим параметрам).
                % Лучшие параметры сохраняются через SaveBestParameters при улучшении,
                % восстанавливаются только в конце обучения.

                % --- Структурный поиск (если включен и не сошёлся) ---
                if obj.TrainingOptions.EnableStructuralSearch && ...
                   ~obj.structuralSearchConverged && ...
                   mod(epoch, obj.TrainingOptions.StructuralSearchInterval) == 0
                    fprintf('\n[Структурная оптимизация] Поиск оптимальной топологии (эпоха %d)...\n', epoch);
                    topologyChanged = obj.StructuralSearchStep(XDataTrain, YDataTrain, XDataTest, YDataTest);
                    obj.UpdateStructuralPlot(ax5);
                    obj.graph.DrawGraph_New([], ax6);
                    if topologyChanged
                        obj.nodes = obj.graph.ListOfNodes;
                        topologyChangedEpochs(end+1) = epoch;
                        c_no = 0;
                        fprintf('[Структурная оптимизация] Топология изменена.\n');
                    end
                end
            end

            % Зачистка бесполезных рёбер после конвергенции (однократно)
            if obj.structuralSearchConverged && ~obj.cleanupDone
                obj.CleanupRedundantEdges(XDataTest, YDataTest);
                obj.cleanupDone = true;
            end

            % Восстанавливаем лучшие параметры (в т.ч. после ранней остановки)
            obj.RestoreBestParameters();

            % Финальное обновление всех графиков
            plot(ax1, 1:epoch, obj.trainErrors, 'b-', 'LineWidth', 1.5);
            hold(ax1, 'on');
            plot(ax1, 1:epoch, obj.testErrors, 'r-', 'LineWidth', 1.5);
            yLimits = ylim(ax1);
            for e = topologyChangedEpochs
                plot(ax1, [e e], yLimits, 'k--', 'LineWidth', 0.7, 'HandleVisibility', 'off');
            end
            hold(ax1, 'off');
            title(ax1, 'Ошибки настройки и тестирования');
            xlabel(ax1, 'Итерация');
            ylabel(ax1, sprintf('%s', obj.TrainingOptions.ErrorMetric));
            legend(ax1, {'Train', 'Test'}, 'Location', 'best');
            grid(ax1, 'on');

            plot(ax3, 1:epoch, epochTimes(1:epoch), 'g-', 'LineWidth', 1.5);
            title(ax3, 'Время расчета эпохи');
            xlabel(ax3, 'Итерация');
            ylabel(ax3, 'Время (сек)');
            grid(ax3, 'on');

            plot(ax4, 1:epoch, errorDiffs(1:epoch), 'm-', 'LineWidth', 1.5);
            hold(ax4, 'on');
            plot(ax4, [1 epoch], [0 0], 'k--', 'LineWidth', 1);
            hold(ax4, 'off');
            title(ax4, '\Delta E (test - train)');
            xlabel(ax4, 'Итерация');
            ylabel(ax4, '\Delta');
            grid(ax4, 'on');

            obj.UpdateStructuralPlot(ax5);
            obj.graph.DrawGraph_New([], ax6);

            sgtitle(sprintf('Обучение завершено (эпоха %d). Лучшая %s: %.4f', ...
                epoch, obj.TrainingOptions.ErrorMetric, obj.bestTestError));
            drawnow;
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
            % Rp-оператор (формула 3.12): случайное смещение параметров на ±RpShiftPercent%
            pct = obj.TrainingOptions.RpShiftPercent / 100;
            obj.nodes = obj.graph.ListOfNodes;
            for i = 1:numel(obj.nodes)
                node = obj.nodes(i);
                edges = node.getOutEdges();
                for j = 1:numel(edges)
                    edge = edges(j);
                    edge.Alfa = edge.Alfa * (1 + (rand() * 2 - 1) * pct);
                    edge.Beta = edge.Beta * (1 + (rand() * 2 - 1) * pct);
                end
                node.Gamma = node.Gamma * (1 + (rand() * 2 - 1) * pct);
            end
        end

        function errorValue = CalculateError(obj, XData, YData, whiteNodeIndices, errorMetric)
            % Вычисляет метрику на тестовой выборке.
            % Первый вызов Forward строит кеш M⁻¹; остальные — O(n²).
            % Поддерживает: mae, mse, rmse, mape.

            if nargin < 4 || isempty(whiteNodeIndices)
                whiteNodeIndices = obj.graph.GetWhiteNodesIndices();
            end
            if nargin < 5 || isempty(errorMetric)
                errorMetric = 'mae';
            end

            validMetrics = {'mae', 'mse', 'rmse', 'mape'};
            metric = lower(errorMetric);
            if ~any(strcmpi(metric, validMetrics))
                error('Недопустимая метрика: %s. Допустимы: %s', metric, strjoin(validMetrics, ', '));
            end

            if length(XData) ~= length(YData)
                error('Размеры XData и YData должны совпадать');
            end
            if isempty(whiteNodeIndices)
                errorValue = NaN; return;
            end

            allWhiteIndices = obj.graph.GetWhiteNodesIndices();
            [~, loc] = ismember(whiteNodeIndices, allWhiteIndices);
            numSamples = length(XData);
            nWhite = length(whiteNodeIndices);
            totalPoints = numSamples * nWhite;

            % Предвыделяем массивы для векторного вычисления
            allPred = zeros(totalPoints, 1);
            allTrue = zeros(totalPoints, 1);

            idx = 1;
            for i = 1:numSamples
                % Быстрый прямой проход (M⁻¹ кеширован, O(n²) вместо O(n³))
                obj.graph.Forward(XData(i));
                pred = obj.graph.GetModelResults();
                selectedPred = pred(whiteNodeIndices);
                selectedTrue = YData(i).getRow(1);
                selectedTrue = selectedTrue(loc);

                for w = 1:nWhite
                    allPred(idx) = selectedPred(w);
                    allTrue(idx) = selectedTrue(w);
                    idx = idx + 1;
                end
            end

            errors = allPred - allTrue;

            switch metric
                case 'mae'
                    errorValue = mean(abs(errors));
                case 'mse'
                    errorValue = mean(errors.^2);
                case 'rmse'
                    errorValue = sqrt(mean(errors.^2));
                case 'mape'
                    epsVal = 1e-10;
                    absTrue = abs(allTrue);
                    absTrue(absTrue < epsVal) = epsVal;
                    errorValue = 100 * mean(abs(errors) ./ absTrue);
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

            % --- Разрешение раздельных границ клиппинга ---
            % Приоритет: 1) откалиброванные, 2) специфичные из TrainingOptions, 3) общие ClipUp/ClipDown
            if obj.clipAutoCalibrated
                clipUpAl = obj.calibratedClipUpAl; clipDownAl = obj.calibratedClipDownAl;
                clipUpBt = obj.calibratedClipUpBt; clipDownBt = obj.calibratedClipDownBt;
                clipUpGm = obj.calibratedClipUpGm; clipDownGm = obj.calibratedClipDownGm;
            else
                clipUpAl = obj.TrainingOptions.ClipUp_Alpha;   if isempty(clipUpAl), clipUpAl = obj.TrainingOptions.ClipUp; end
                clipDownAl = obj.TrainingOptions.ClipDown_Alpha; if isempty(clipDownAl), clipDownAl = obj.TrainingOptions.ClipDown; end
                clipUpBt = obj.TrainingOptions.ClipUp_Beta;     if isempty(clipUpBt), clipUpBt = obj.TrainingOptions.ClipUp; end
                clipDownBt = obj.TrainingOptions.ClipDown_Beta;  if isempty(clipDownBt), clipDownBt = obj.TrainingOptions.ClipDown; end
                clipUpGm = obj.TrainingOptions.ClipUp_Gamma;    if isempty(clipUpGm), clipUpGm = obj.TrainingOptions.ClipUp; end
                clipDownGm = obj.TrainingOptions.ClipDown_Gamma; if isempty(clipDownGm), clipDownGm = obj.TrainingOptions.ClipDown; end
            end

            % Инициализация градиентов по всем батчам
            batchAlGrad = cell(numNodes, numBatches);
            batchBtGrad = cell(numNodes, numBatches);
            batchGmGrad = cell(numNodes, numBatches);

            % Инициализация дельта-массивов для всего батча
            delta_in_cache = cell(numNodes, numBatches);
            delta_out_cache = cell(numNodes, numBatches);

            for batchIdx = 1:numBatches
                obj.t = obj.t + 1;
                batchStart = (batchIdx-1)*obj.TrainingOptions.BatchSize + 1;
                batchEnd = min(batchIdx*obj.TrainingOptions.BatchSize, numSamples);
                batchIndices = batchStart:batchEnd;
                numInBatch = length(batchIndices);

                obj.updateProgress(batchIdx, numBatches);

                % Вычисляем delta_in, delta_out для всего батча
                % Вычисляем все суммы в знаменателе
                sum_alpha_out_plus_one = zeros(1, numNodes);

                for i = 1:numNodes
                    outgoingEdges = obj.outgoingEdgesCache{i};
                    if ~isempty(outgoingEdges)
                        % Векторизованное вычисление суммы
                        alphas = [outgoingEdges.Alfa];
                        sum_alpha_out_plus_one(i) = sum(alphas) + 1;
                    else
                        sum_alpha_out_plus_one(i) = 1; % 1 для пустого набора
                    end
                end
                
                % Основной цикл
                for i = 1:numNodes
                    incomingEdges = obj.incomingEdgesCache{i};
                    outgoingEdges = obj.outgoingEdgesCache{i};

                    delta_in = zeros(1, numNodes);
                    if ~isempty(incomingEdges)
                        for edge_idx = 1:numel(incomingEdges)
                            e = incomingEdges(edge_idx);
                            sourceNode = e.SourceNode;
                            sourceIdx = sourceNode.ID;
                            if ~isempty(sourceIdx) && sourceIdx > 0
                                % Нормируем на знаменатель ИСТОЧНИКА (sourceIdx)
                                delta_in(sourceIdx) = e.Alfa / sum_alpha_out_plus_one(sourceIdx);
                            end
                        end
                    end
                    delta_in_cache{i,batchIdx} = delta_in;

                    delta_out = zeros(1, numNodes);
                    if ~isempty(outgoingEdges)
                        for edge_idx = 1:numel(outgoingEdges)
                            e = outgoingEdges(edge_idx);
                            targetNode = e.TargetNode;
                            targetIdx = targetNode.ID;
                            if ~isempty(targetIdx) && targetIdx > 0
                                delta_out(targetIdx) = e.Alfa / sum_alpha_out_plus_one(i);
                            end
                        end
                    end
                    delta_out_cache{i,batchIdx} = delta_out;
                end

                for i = 1:numNodes
                    numEdges = numel(obj.outgoingEdgesCache{i});
                    batchAlGrad{i,batchIdx} = zeros(1, numEdges);
                    batchBtGrad{i,batchIdx} = zeros(1, numEdges);
                    batchGmGrad{i,batchIdx} = 0;
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

                    % J_self для черных вершин
                    J_self = zeros(1, numNodes);
                    for b = obj.blackNodeIndices
                        outgoingEdges = obj.outgoingEdgesCache{b};

                        % Знаменатель из (2.6): 1 + Σα_out
                        denominator = 1 + sum([outgoingEdges.Alfa]);

                        % G_In — вклад входящих соседей (формула 2.3)
                        G_in = 0;
                        incomingEdges = obj.incomingEdgesCache{b};
                        for e_idx = 1:numel(incomingEdges)
                            e = incomingEdges(e_idx);
                            sourceNode = e.SourceNode;
                            sourceIdx = sourceNode.ID;
                            G_in = G_in + e.Alfa * modelValues(sourceIdx) + e.Beta;
                        end

                        % Σβ_out
                        sum_beta_out = sum([outgoingEdges.Beta]);

                        % F_shadow = та же формула (2.6), но без L_b — вклад только от соседей
                        F_shadow = (G_in - sum_beta_out) / denominator;

                        F_b = modelValues(b);

                        % Собственная невязка: J_self = L_b / denominator
                        J_self(b) = (F_b - F_shadow);
                    end

                    J_total = zeros(1, numNodes);
                    J_total(obj.whiteNodeIndices) = J_white(obj.whiteNodeIndices);

                    max_iterations = length(obj.blackNodeIndices) + 1;

                    A_in = eye(numNodes,numNodes);
                    A_out = eye(numNodes,numNodes);

                    for iter = 1:max_iterations
                        J_prev = J_total;
                        J_new = J_total;
                        updated = false;

                        for i = obj.blackNodeIndices
                            delta_in = delta_in_cache{i,batchIdx};
                            delta_out = delta_out_cache{i,batchIdx};

                            incomingNeighbors = obj.incomingNeighborsCache{i};
                            outgoingEdges = obj.outgoingEdgesCache{i}';

                            % Сумма по входящим соседям
                            sum_in = 0;
                            for neighbor = incomingNeighbors
                                if ~isempty(neighbor)
                                    neighborIdx = neighbor.ID;
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
                                targetIdx = targetNode.ID;
                                if ~isempty(targetIdx) && delta_out(targetIdx) ~= 0
                                    if A_out(i, targetIdx) == 0 && J_prev(targetIdx) ~= 0
                                        sum_out = sum_out + delta_out(targetIdx) * J_prev(targetIdx);
                                        A_out(i, targetIdx) = 1;
                                    end
                                end
                            end

                            new_value = obj.TrainingOptions.Lambda_Self * J_self(i) ...
                                + obj.TrainingOptions.Lambda_Struct * (sum_in + sum_out);

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

                    % --- Настройка ядровых функций (ITunableCoreF) ---
                    % dJ/dC = J_total(i) / D(i,i), где D(i,i) = Σ(α_out + 1)
                    % Не привязан к конкретным параметрам — работает с любой ITunableCoreF
                    for i = 1:numNodes
                        if J_total(i) == 0, continue; end

                        nodeFunc = obj.nodes(i).getNodeFunction();
                        if isempty(nodeFunc) || ~isa(nodeFunc, 'coreFunctions.ITunableCoreF')
                            continue;
                        end

                        outgoingEdges = obj.outgoingEdgesCache{i};
                        if ~isempty(outgoingEdges)
                            denominator = 1 + sum([outgoingEdges.Alfa]);
                        else
                            denominator = 1;
                        end

                        dJ_dC = J_total(i) / denominator;
                        nodeInputData = xMatrix.getRow(i);
                        nodeFunc.TuneParameters(nodeInputData, dJ_dC);
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
                        h_i = obj.TrainingOptions.getNodeMultiplier(i);
                        batchGmGrad{i} = batchGmGrad{i} - h_i * dF_dgamma * J_total(i);
                        
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

                            % Обновление градиентов (формула 3.10-3.11) с h_v
                            batchAlGrad{i,batchIdx}(j) = batchAlGrad{i,batchIdx}(j) - h_i * dF_dalpha_out * J_total(i);
                            batchBtGrad{i,batchIdx}(j) = batchBtGrad{i,batchIdx}(j) - h_i * dF_dbeta_out * J_total(i);
                        end

                        % Градиенты для входящих рёбер (alpha и beta)
                        incomingEdges = obj.incomingEdgesCache{i};
                        for j = 1:numel(incomingEdges)
                            edge = incomingEdges(j);
                            sourceNode = edge.SourceNode;
                            sourceIdx = sourceNode.ID;

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
                                    batchAlGrad{sourceIdx,batchIdx}(edgePos) = batchAlGrad{sourceIdx,batchIdx}(edgePos) - ...
                                        h_i * dF_dalpha_in * J_total(i);
                                    batchBtGrad{sourceIdx,batchIdx}(edgePos) = batchBtGrad{sourceIdx,batchIdx}(edgePos) - ...
                                        h_i * dF_dbeta_in * J_total(i);
                                end
                            end
                        end
                    end

                    % Добавляем L2 регуляризацию (λ1 в формуле 3.7)
                    for i = 1:numNodes
                        % Регуляризация по gamma
                        batchGmGrad{i} = batchGmGrad{i} + obj.TrainingOptions.Lambda_Gamma * obj.nodes(i).Gamma;
                        edges = obj.outgoingEdgesCache{i};
                        for j = 1:numel(edges)
                            edge = edges(j);
                            batchAlGrad{i,batchIdx}(j) = batchAlGrad{i,batchIdx}(j) + obj.TrainingOptions.Lambda_Alph * edge.Alfa;
                            batchBtGrad{i,batchIdx}(j) = batchBtGrad{i,batchIdx}(j) + obj.TrainingOptions.Lambda_Beta * edge.Beta;
                        end
                    end
                end

                % --- Стабилизирующий регуляризатор Rs(α) — формулы (3.5), (3.9a)-(3.9c) ---
                lambda_Rs = obj.TrainingOptions.Lambda_Stability;
                if ~isempty(lambda_Rs) && lambda_Rs > 0
                    for i = 1:numNodes
                        edges_i = obj.outgoingEdgesCache{i};
                        if isempty(edges_i) && isempty(obj.incomingEdgesCache{i})
                            continue;
                        end

                        D_i = 1 + sum([edges_i.Alfa]);

                        sum_alpha_in = 0;
                        incomingEdges = obj.incomingEdgesCache{i};
                        for e_idx = 1:numel(incomingEdges)
                            sum_alpha_in = sum_alpha_in + incomingEdges(e_idx).Alfa;
                        end

                        r_v = max(0, sum_alpha_in / D_i - 1);  % формула (3.11)
                        if r_v <= 0, continue; end

                        coef = lambda_Rs * 2 * r_v;

                        % Производная по входящим α: +2·r_v / D(v) — формула (3.9c)
                        for e_idx = 1:numel(incomingEdges)
                            e = incomingEdges(e_idx);
                            sourceIdx = e.SourceNode.ID;
                            if isempty(sourceIdx), continue; end
                            sourceEdges = obj.outgoingEdgesCache{sourceIdx};
                            edgePos = find(sourceEdges == e, 1);
                            if ~isempty(edgePos)
                                batchAlGrad{sourceIdx,batchIdx}(edgePos) = ...
                                    batchAlGrad{sourceIdx,batchIdx}(edgePos) + coef / D_i;
                            end
                        end

                        % Производная по исходящим α: -2·r_v · Σα_in / D(v)² — формула (3.9a)
                        if ~isempty(edges_i)
                            dRs_out = -coef * sum_alpha_in / (D_i^2);
                            batchAlGrad{i,batchIdx} = batchAlGrad{i,batchIdx} + dRs_out;
                        end
                    end
                end

                % --- Нормализация градиентов ---
                invNumInBatch = 1 / numInBatch;
                for i = 1:numNodes
                    if ~isempty(batchAlGrad{i,batchIdx})
                        batchAlGrad{i,batchIdx} = batchAlGrad{i,batchIdx} * invNumInBatch;
                    end
                    if ~isempty(batchBtGrad{i,batchIdx})
                        batchBtGrad{i,batchIdx} = batchBtGrad{i,batchIdx} * invNumInBatch;
                    end
                    if ~isempty(batchGmGrad{i,batchIdx})
                        batchGmGrad{i,batchIdx} = batchGmGrad{i,batchIdx} * invNumInBatch;
                    end
                end

                % --- Авто-калибровка клиппинга по первому батчу ---
                if obj.TrainingOptions.AutoCalibrateClip && ~obj.clipAutoCalibrated
                    allAl = []; allBt = []; allGm = [];
                    for i = 1:numNodes
                        if ~isempty(batchAlGrad{i,batchIdx}), allAl = [allAl, abs(batchAlGrad{i,batchIdx}(:))']; end
                        if ~isempty(batchBtGrad{i,batchIdx}), allBt = [allBt, abs(batchBtGrad{i,batchIdx}(:))']; end
                        if ~isempty(batchGmGrad{i,batchIdx}), allGm = [allGm, abs(batchGmGrad{i,batchIdx}(:))']; end
                    end
                    pct = obj.TrainingOptions.ClipPercentile;
                    cal = @(g) max(prctile(g, pct), eps);
                    if isempty(allAl), obj.calibratedClipUpAl = 1; else, obj.calibratedClipUpAl = cal(allAl); end
                    if isempty(allBt), obj.calibratedClipUpBt = 1; else, obj.calibratedClipUpBt = cal(allBt); end
                    if isempty(allGm), obj.calibratedClipUpGm = 1; else, obj.calibratedClipUpGm = cal(allGm); end
                    obj.calibratedClipDownAl = -obj.calibratedClipUpAl;
                    obj.calibratedClipDownBt = -obj.calibratedClipUpBt;
                    obj.calibratedClipDownGm = -obj.calibratedClipUpGm;
                    obj.clipAutoCalibrated = true;

                    % Обновляем эффективные границы
                    clipUpAl = obj.calibratedClipUpAl; clipDownAl = obj.calibratedClipDownAl;
                    clipUpBt = obj.calibratedClipUpBt; clipDownBt = obj.calibratedClipDownBt;
                    clipUpGm = obj.calibratedClipUpGm; clipDownGm = obj.calibratedClipDownGm;

                    fprintf('\nАвто-калибровка клиппинга (P%d): α=±%.2e, β=±%.2e, γ=±%.2e\n', ...
                        pct, clipUpAl, clipUpBt, clipUpGm);
                end

                % --- Применение клиппинга ---
                for i = 1:numNodes
                    if ~isempty(batchAlGrad{i,batchIdx})
                        batchAlGrad{i,batchIdx} = min(max(batchAlGrad{i,batchIdx}, clipDownAl), clipUpAl);
                    end
                    if ~isempty(batchBtGrad{i,batchIdx})
                        batchBtGrad{i,batchIdx} = min(max(batchBtGrad{i,batchIdx}, clipDownBt), clipUpBt);
                    end
                    if ~isempty(batchGmGrad{i,batchIdx})
                        batchGmGrad{i,batchIdx} = min(max(batchGmGrad{i,batchIdx}, clipDownGm), clipUpGm);
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
                    obj.mAl{i} = obj.TrainingOptions.Beta1 * obj.mAl{i} + (1-obj.TrainingOptions.Beta1) * batchAlGrad{i,batchIdx};
                    obj.vAl{i} = obj.TrainingOptions.Beta2 * obj.vAl{i} + (1-obj.TrainingOptions.Beta2) * (batchAlGrad{i,batchIdx}.^2);
                    obj.mBt{i} = obj.TrainingOptions.Beta1 * obj.mBt{i} + (1-obj.TrainingOptions.Beta1) * batchBtGrad{i,batchIdx};
                    obj.vBt{i} = obj.TrainingOptions.Beta2 * obj.vBt{i} + (1-obj.TrainingOptions.Beta2) * (batchBtGrad{i,batchIdx}.^2);
                    obj.mGm{i} = obj.TrainingOptions.Beta1 * obj.mGm{i} + (1-obj.TrainingOptions.Beta1) * batchGmGrad{i,batchIdx};
                    obj.vGm{i} = obj.TrainingOptions.Beta2 * obj.vGm{i} + (1-obj.TrainingOptions.Beta2) * (batchGmGrad{i,batchIdx}.^2);

                    % Применение обновлений (формула 3.16)
                    lr = obj.TrainingOptions.LearningRate;
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

        function topologyChanged = StructuralSearchStep(obj, XDataTrain, YDataTrain, XDataTest, YDataTest)
            % Жадный поиск оптимальной топологии: перебирает случайные мутации,
            % быстро обучает кандидатов и выбирает лучшего
            topologyChanged = false;
            obj.structuralSearchStepCount = obj.structuralSearchStepCount + 1;

            opts = obj.TrainingOptions;

            possibleEdges = obj.graph.getPossibleEdges();
            existingEdges = obj.graph.getExistingEdges();
            currentCount = obj.graph.getTotalEdgeCount();

            canAdd = ~isempty(possibleEdges) && currentCount < opts.StructuralSearchMaxEdges;
            canRemove = ~isempty(existingEdges) && currentCount > opts.StructuralSearchMinEdges;

            if ~canAdd && ~canRemove
                fprintf('[Структурная оптимизация] Нет допустимых мутаций (границы плотности).\n');
                return;
            end

            % Сохраняем состояние исходного графа
            savedGraph = obj.graph;
            savedNodes = obj.nodes;
            savedBestAl = obj.bestAl;
            savedBestBt = obj.bestBt;
            savedBestGm = obj.bestGm;
            savedTrainErrors = obj.trainErrors;
            savedTestErrors = obj.testErrors;
            savedErrorArray = obj.errorArray;

            baselineError = obj.bestTestError;  % фиксированный порог для ВСЕХ кандидатов
            bestCandidateError = Inf;
            bestMutation = [];

            % ===== Хеш текущей топологии для инвалидации глобального кеша =====
            if isempty(existingEdges)
                graphHash = 'no_edges';
            else
                parts = cell(1, size(existingEdges, 1));
                sorted = sortrows(existingEdges, [1 2]);
                for r = 1:size(sorted, 1)
                    parts{r} = sprintf('%d->%d', sorted(r,1), sorted(r,2));
                end
                graphHash = strjoin(parts, '|');
            end

            % Инициализируем глобальный кеш при первом вызове
            if isempty(obj.globalEdgeCache)
                obj.globalEdgeCache = containers.Map('KeyType', 'char', 'ValueType', 'any');
            end

            % ===== Формируем пул мутаций (приоритет — непроверенным рёбрам) =====
            % 1. Разделяем possibleEdges на непроверенные и проверенные
            if canAdd
                untestedAdd = []; testedAdd = [];
                for k = 1:size(possibleEdges, 1)
                    edgeKey = sprintf('%d->%d', possibleEdges(k,1), possibleEdges(k,2));
                    if isKey(obj.globalEdgeCache, edgeKey)
                        cached = obj.globalEdgeCache(edgeKey);
                        % Совпал хеш → уже проверено при той же топологии
                        if strcmp(cached.graphHash, graphHash)
                            testedAdd(end+1, :) = possibleEdges(k, :);
                            continue;
                        end
                        % Cooldown: ребро недавно удалено как улучшение → не проверяем
                        if isfield(cached, 'removedInStep') && cached.removedInStep > 0 ...
                                && obj.structuralSearchStepCount - cached.removedInStep <= obj.TrainingOptions.StructuralCooldown
                            testedAdd(end+1, :) = possibleEdges(k, :);
                            continue;
                        end
                    end
                    untestedAdd(end+1, :) = possibleEdges(k, :);  % не проверено или топология изменилась
                end
                % Перемешиваем
                if ~isempty(untestedAdd), untestedAdd = untestedAdd(randperm(size(untestedAdd,1)), :); end
                if ~isempty(testedAdd),   testedAdd   = testedAdd(randperm(size(testedAdd,1)), :); end
            end

            % 2. Аналогично для remove
            if canRemove
                untestedRemove = []; testedRemove = [];
                for k = 1:size(existingEdges, 1)
                    edgeKey = sprintf('%d->%d', existingEdges(k,1), existingEdges(k,2));
                    if isKey(obj.globalEdgeCache, edgeKey)
                        cached = obj.globalEdgeCache(edgeKey);
                        if strcmp(cached.graphHash, graphHash)
                            testedRemove(end+1, :) = existingEdges(k, :);
                            continue;
                        end
                    end
                    untestedRemove(end+1, :) = existingEdges(k, :);
                end
                if ~isempty(untestedRemove), untestedRemove = untestedRemove(randperm(size(untestedRemove,1)), :); end
                if ~isempty(testedRemove),   testedRemove   = testedRemove(randperm(size(testedRemove,1)), :); end
            end

            % 3. Случайная очередь: непроверенные ×2 (приоритет), проверенные ×1
            mutationPool = {};
            if canAdd
                for k = 1:size(untestedAdd, 1)
                    mutationPool{end+1} = struct('type', 'add', 'src', untestedAdd(k,1), 'dst', untestedAdd(k,2), 'prio', 1);
                end
                for k = 1:size(testedAdd, 1)
                    mutationPool{end+1} = struct('type', 'add', 'src', testedAdd(k,1), 'dst', testedAdd(k,2), 'prio', 0);
                end
            end
            if canRemove
                for k = 1:size(untestedRemove, 1)
                    mutationPool{end+1} = struct('type', 'remove', 'src', untestedRemove(k,1), 'dst', untestedRemove(k,2), 'prio', 1);
                end
                for k = 1:size(testedRemove, 1)
                    mutationPool{end+1} = struct('type', 'remove', 'src', testedRemove(k,1), 'dst', testedRemove(k,2), 'prio', 0);
                end
            end

            % Взвешенная случайная выборка без повторений: непроверенные ×2
            if ~isempty(mutationPool)
                % Дублируем приоритетные → выбор без повторений, но с перевесом
                weightedPool = {};
                for k = 1:numel(mutationPool)
                    weightedPool{end+1} = mutationPool{k};
                    if mutationPool{k}.prio, weightedPool{end+1} = mutationPool{k}; end
                end
                maxCandidates = min(opts.StructuralSearchCandidates, numel(mutationPool));
                n = min(maxCandidates, numel(weightedPool));
                idx = randperm(numel(weightedPool), n);
                % Убираем дубликаты (могли выбрать две копии одного кандидата)
                seen = containers.Map('KeyType', 'char', 'ValueType', 'logical');
                mutationQueue = {};
                for k = 1:n
                    m = weightedPool{idx(k)};
                    key = sprintf('%s:%d->%d', m.type, m.src, m.dst);
                    if ~isKey(seen, key)
                        seen(key) = true;
                        mutationQueue{end+1} = m;
                    end
                end
            else
                mutationQueue = {};
            end

            if isempty(mutationQueue)
                fprintf('[Структурная оптимизация] Все кандидаты проверены — оптимальная структура найдена.\n');
                obj.structuralSearchConverged = true;
                return;
            end

            fprintf('[Структурная оптимизация] Проверка %d кандидатов (add=%d, remove=%d, reconnect=%d)...\n', ...
                numel(mutationQueue), ...
                sum(cellfun(@(m) strcmp(m.type,'add'), mutationQueue)), ...
                sum(cellfun(@(m) strcmp(m.type,'remove'), mutationQueue)), ...
                sum(cellfun(@(m) strcmp(m.type,'reconnect'), mutationQueue)));

            % Кеш проверенных рёбер в рамках одного шага поиска
            checkedEdges = containers.Map('KeyType', 'char', 'ValueType', 'double');
            obj.rejectedEdges = zeros(0, 2);
            skippedCount = 0;

            for c = 1:numel(mutationQueue)
                mutation = mutationQueue{c};

                % Проверка: не тестируем одно и то же ребро дважды за шаг
                cacheKey = sprintf('%s:%d->%d', mutation.type, mutation.src, mutation.dst);
                if isKey(checkedEdges, cacheKey)
                    skippedCount = skippedCount + 1;
                    fprintf('  [%d/%d] %s: %d->%d — пропущен (дубликат)\n', ...
                        c, numel(mutationQueue), mutation.type, mutation.src, mutation.dst);
                    continue;
                end

                % Клонируем граф и применяем мутацию
                candidateGraph = savedGraph.clone();
                switch mutation.type
                    case 'add'
                        candidateGraph.addEdgeBetween(mutation.src, mutation.dst, ...
                            opts.StructInitAlpha, opts.StructInitBeta);
                    case 'remove'
                        candidateGraph.removeEdgeBetween(mutation.src, mutation.dst);
                    case 'reconnect'
                        candidateGraph.removeEdgeBetween(mutation.src, mutation.dst);
                        candidateGraph.addEdgeBetween(mutation.src2, mutation.dst2, ...
                            opts.StructInitAlpha, opts.StructInitBeta);
                end

                % Проверяем устойчивость кандидата (условие 3.3 из рукописи)
                [isStable, ~] = candidateGraph.checkStability();
                if ~isStable
                    % Кешируем неустойчивый результат, чтобы не перепроверять
                    fprintf('  [%d/%d] %s: %d->%d — неустойчив, пропущен\n', ...
                        c, numel(mutationQueue), mutation.type, mutation.src, mutation.dst);
                    checkedEdges(cacheKey) = Inf;
                    if strcmp(mutation.type, 'add') || strcmp(mutation.type, 'remove')
                        edgeKey = sprintf('%d->%d', mutation.src, mutation.dst);
                        obj.globalEdgeCache(edgeKey) = struct('error', Inf, 'graphHash', graphHash, 'removedInStep', 0);
                    end
                    continue;
                end

                % Быстрое обучение кандидата
                fprintf('  [%d/%d] %s: %d->%d ... ', ...
                    c, numel(mutationQueue), mutation.type, mutation.src, mutation.dst);
                obj.graph = candidateGraph;
                obj.nodes = candidateGraph.ListOfNodes;
                obj.ResetTrainingState();

                candidateError = obj.QuickEvaluate(XDataTrain, YDataTrain, ...
                    XDataTest, YDataTest, opts.StructuralSearchEpochs);

                checkedEdges(cacheKey) = candidateError;

                % Сохраняем в глобальный кеш (ключ — ребро, значение — ошибка + хеш топологии)
                if strcmp(mutation.type, 'add') || strcmp(mutation.type, 'remove')
                    edgeKey = sprintf('%d->%d', mutation.src, mutation.dst);
                    obj.globalEdgeCache(edgeKey) = struct('error', candidateError, 'graphHash', graphHash, 'removedInStep', 0);
                end

                % Штраф за сложность: добавление ребра требует значимого улучшения
                deltaEdges = 0;
                if strcmp(mutation.type, 'add'), deltaEdges = 1;
                elseif strcmp(mutation.type, 'remove'), deltaEdges = -1;
                end
                penalizedError = candidateError * (1 + opts.StructuralComplexityPenalty * deltaEdges);

                if penalizedError < baselineError && candidateError < bestCandidateError
                    bestCandidateError = candidateError;
                    bestMutation = mutation;
                    fprintf('MAE=%.4f ✓ (улучшение)\n', candidateError);
                else
                    fprintf('MAE=%.4f ✗\n', candidateError);
                    % Отклонённое ребро — сохраняем для визуализации
                    obj.rejectedEdges(end+1, :) = [mutation.src, mutation.dst];
                end
            end

            if skippedCount > 0
                fprintf('[Структурная оптимизация] Пропущено дубликатов: %d\n', skippedCount);
            end

            % Если улучшений нет и все рёбра проверены — структура оптимальна
            if isempty(bestMutation) && canAdd
                allCovered = true;
                for k = 1:size(possibleEdges, 1)
                    edgeKey = sprintf('%d->%d', possibleEdges(k,1), possibleEdges(k,2));
                    if ~isKey(obj.globalEdgeCache, edgeKey)
                        allCovered = false; break;
                    end
                    cached = obj.globalEdgeCache(edgeKey);
                    if ~strcmp(cached.graphHash, graphHash)
                        allCovered = false; break;
                    end
                end
                if allCovered
                    fprintf('[Структурная оптимизация] Все комбинации проверены — оптимальная структура найдена.\n');
                    obj.structuralSearchConverged = true;
                end
            end

            % Восстанавливаем исходный граф и состояние
            obj.graph = savedGraph;
            obj.nodes = savedNodes;
            obj.bestAl = savedBestAl;
            obj.bestBt = savedBestBt;
            obj.bestGm = savedBestGm;
            obj.trainErrors = savedTrainErrors;
            obj.testErrors = savedTestErrors;
            obj.errorArray = savedErrorArray;
            obj.ResetTrainingState();

            % Применяем лучшую мутацию к реальному графу
            if ~isempty(bestMutation)
                switch bestMutation.type
                    case 'add'
                        obj.graph.addEdgeBetween(bestMutation.src, bestMutation.dst, ...
                            opts.StructInitAlpha, opts.StructInitBeta);
                        fprintf('[Структурная оптимизация] Добавлено ребро %d->%d\n', bestMutation.src, bestMutation.dst);
                    case 'remove'
                        obj.graph.removeEdgeBetween(bestMutation.src, bestMutation.dst);
                        fprintf('[Структурная оптимизация] Удалено ребро %d->%d\n', bestMutation.src, bestMutation.dst);
                    case 'reconnect'
                        obj.graph.removeEdgeBetween(bestMutation.src, bestMutation.dst);
                        obj.graph.addEdgeBetween(bestMutation.src2, bestMutation.dst2, ...
                            opts.StructInitAlpha, opts.StructInitBeta);
                        fprintf('[Структурная оптимизация] Переподключено: %d->%d → %d->%d\n', ...
                            bestMutation.src, bestMutation.dst, bestMutation.src2, bestMutation.dst2);
                end

                % Cooldown: если удалили ребро — не добавлять его обратно 2 шага
                if strcmp(bestMutation.type, 'remove')
                    edgeKey = sprintf('%d->%d', bestMutation.src, bestMutation.dst);
                    if isKey(obj.globalEdgeCache, edgeKey)
                        cached = obj.globalEdgeCache(edgeKey);
                        cached.removedInStep = obj.structuralSearchStepCount;
                        obj.globalEdgeCache(edgeKey) = cached;
                    end
                end

                obj.nodes = obj.graph.ListOfNodes;
                obj.bestTestError = bestCandidateError;
                obj.SaveBestParameters();
                obj.ResetTrainingStateForNewEdge();
                obj.clipAutoCalibrated = false;
                topologyChanged = true;

                fprintf('[Структурная оптимизация] Топология улучшена! Ошибка: %.4f (рёбер: %d)\n', ...
                    bestCandidateError, obj.graph.getTotalEdgeCount());
            else
                fprintf('[Структурная оптимизация] Улучшений не найдено (порог: %.4f).\n', baselineError);
            end
        end

        function bestError = QuickEvaluate(obj, XDataTrain, YDataTrain, XDataTest, YDataTest, numEpochs)
            % Быстрое обучение: N эпох, возвращает среднюю ошибку по всем эпохам
            sumErrors = 0;
            for ep = 1:numEpochs
                obj.Compute_V5(XDataTrain, YDataTrain);
                sumErrors = sumErrors + obj.CalculateError(XDataTest, YDataTest, ...
                    obj.TrainingOptions.TargetNodeIndices, obj.TrainingOptions.ErrorMetric);
            end
            bestError = sumErrors / numEpochs;
        end

        function CleanupRedundantEdges(obj, XDataTest, YDataTest)
            % Удаляет рёбра, не влияющие на ошибку (Δ < порог)
            threshold = obj.TrainingOptions.StructuralCleanupThreshold;
            if threshold == 0, return; end

            whiteNodes = obj.graph.GetWhiteNodesIndices();
            baseline = obj.CalculateError(XDataTest, YDataTest, whiteNodes, ...
                obj.TrainingOptions.ErrorMetric);

            fprintf('\n[Зачистка] Проверка %d рёбер (порог: %.1f%%, baseline: %.4f)\n', ...
                obj.graph.getTotalEdgeCount(), threshold*100, baseline);

            edges = obj.graph.getExistingEdges();
            removed = 0;
            for k = 1:size(edges, 1)
                src = edges(k, 1); dst = edges(k, 2);
                [a, b] = obj.graph.getEdgeParams(src, dst);

                fprintf('  [%d/%d] %d->%d (α=%.4f, β=%.4f) — ', k, size(edges,1), src, dst, a, b);

                obj.graph.removeEdgeBetween(src, dst);
                err = obj.CalculateError(XDataTest, YDataTest, whiteNodes, ...
                    obj.TrainingOptions.ErrorMetric);
                delta = (err - baseline) / max(1, abs(baseline));

                if delta <= threshold
                    removed = removed + 1;
                    fprintf('УДАЛЕНО (Δ=%.4f%%, ошибка: %.4f)\n', delta*100, err);
                else
                    obj.graph.addEdgeBetween(src, dst, a, b);
                    fprintf('оставлено (Δ=%.4f%%, ошибка: %.4f)\n', delta*100, err);
                end
            end

            if removed > 0
                obj.nodes = obj.graph.ListOfNodes;
                obj.ResetTrainingState();
                fprintf('[Зачистка] Итого: удалено %d рёбер. Ошибка: %.4f → %.4f\n', ...
                    removed, baseline, obj.CalculateError(XDataTest, YDataTest, whiteNodes, obj.TrainingOptions.ErrorMetric));
            else
                fprintf('[Зачистка] Все рёбра значимы (порог: %.1f%%).\n', threshold*100);
            end
        end

        function ResetTrainingState(obj)
            % Сбрасывает кеши и моменты ADAM при изменении топологии
            obj.incomingEdgesCache = {};
            obj.outgoingEdgesCache = {};
            obj.incomingNeighborsCache = {};
            obj.mAl = {};
            obj.vAl = {};
            obj.mBt = {};
            obj.vBt = {};
            obj.mGm = {};
            obj.vGm = {};
            obj.t = 0;
            obj.whiteNodeIndices = [];
            obj.blackNodeIndices = [];
        end

        function ResetTrainingStateForNewEdge(obj)
            % Сброс кешей (структура изменилась), но сохранение моментов ADAM.
            % Моменты достраиваются только для новых рёбер/вершин, старые не трогаем.
            obj.incomingEdgesCache = {};
            obj.outgoingEdgesCache = {};
            obj.incomingNeighborsCache = {};
            obj.whiteNodeIndices = [];
            obj.blackNodeIndices = [];

            numNodes = numel(obj.nodes);
            for i = 1:numNodes
                numEdges = numel(obj.nodes(i).getOutEdges());
                % Gamma-моменты: инициализируем если нет (новая вершина или первый раз)
                if numel(obj.mGm) < i || isempty(obj.mGm{i})
                    obj.mGm{i} = 0;
                    obj.vGm{i} = 0;
                end
                if numEdges == 0
                    continue;
                end
                % Достраиваем моменты для рёбер
                if numel(obj.mAl) < i || isempty(obj.mAl{i})
                    obj.mAl{i} = zeros(1, numEdges);
                    obj.vAl{i} = zeros(1, numEdges);
                    obj.mBt{i} = zeros(1, numEdges);
                    obj.vBt{i} = zeros(1, numEdges);
                else
                    oldLen = numel(obj.mAl{i});
                    if numEdges > oldLen
                        obj.mAl{i}(end+1:numEdges) = 0;
                        obj.vAl{i}(end+1:numEdges) = 0;
                        obj.mBt{i}(end+1:numEdges) = 0;
                        obj.vBt{i}(end+1:numEdges) = 0;
                    end
                end
            end
        end

        function UpdateStructuralPlot(obj, ax)
            % Отображает матрицу смежности: зелёный — рёбра,
            % красный — отклонённые кандидаты, серый — непроверенные

            n = numel(obj.graph.ListOfNodes);
            ids = arrayfun(@(nd) nd.ID, obj.graph.ListOfNodes);

            M = zeros(n, n);
            for i = 1:n
                for j = 1:n
                    if i == j
                        M(i,j) = NaN;
                    elseif obj.graph.hasEdge(ids(i), ids(j))
                        M(i,j) = 1;
                    elseif ~isempty(obj.rejectedEdges)
                        idx = find(obj.rejectedEdges(:,1) == ids(i) & obj.rejectedEdges(:,2) == ids(j), 1);
                        if ~isempty(idx), M(i,j) = -1; end
                    end
                end
            end

            cla(ax);
            imagesc(ax, M, [-1, 1]);
            colormap(ax, [1 0.7 0.7; 0.85 0.85 0.85; 0.4 0.8 0.4]);

            ax.XTick = 1:n; ax.YTick = 1:n;
            ax.XAxisLocation = 'top';
            ax.XTickLabel = arrayfun(@(id) sprintf('%d', id), ids, 'UniformOutput', false);
            ax.YTickLabel = arrayfun(@(id) sprintf('%d', id), ids, 'UniformOutput', false);
            title(ax, 'Структура графа');
            xlabel(ax, '\rightarrow to');
            ylabel(ax, 'from \rightarrow');

            for i = 1:n
                for j = 1:n
                    if isnan(M(i,j)), continue; end
                    if M(i,j) == 1, lbl = '\surd'; c = 'k';
                    elseif M(i,j) == -1, lbl = '\times'; c = 'k';
                    else, lbl = ''; c = 'k';
                    end
                    text(ax, j, i, lbl, 'HorizontalAlignment', 'center', ...
                        'Color', c, 'FontSize', 14, 'FontWeight', 'bold');
                end
            end
        end
    end
end

