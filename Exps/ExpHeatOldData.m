%% ExpHeatOldData.m — Эксперимент с лаговым признаком Tmax на графовой модели
% Архитектура: 3 зональные вершины (1↔2↔3) + 1 лаговая вершина (4→3)
% Лаговая вершина получает Tmax с предыдущего наблюдения
% Использует DataForCompr.xlsx (6098 строк, выраженная автокорреляция Tmax)
clear; clc; close all;
rng(1111);

import BWGraph.*;
import BWGraph.CustomMatrix.*;
import BWGraph.RandomGenerator.*;
import BWGraph.Trainer.*;

%% 1. Загрузка и подготовка данных
fprintf('=== Загрузка DataForCompr.xlsx ===\n');
data = readtable("DataForCompr.xlsx");
data = data(1:150,:);
fprintf('Загружено %d строк, %d столбцов\n', height(data), width(data));

% Лаговый признак — Tmax с предыдущего наблюдения
Tmax_lag = [NaN; data.Tmax(1:end-1)];
fprintf('Лаговый Tmax: %d не-NaN значений\n', sum(~isnan(Tmax_lag)));

% Удаляем первую строку (нет лага)
valid = ~isnan(Tmax_lag);
data = data(valid, :);
Tmax_lag = Tmax_lag(valid);
fprintf('После удаления первой строки: %d строк\n', height(data));

totalBatch = height(data);

%% 2. Формирование признаков для каждой вершины
% Зона 1: [время/100, средняя температура лево-право]
t1 = data.F12_TimeDiff / 100;
T1 = (data.F12_TL + data.F12_TR) / 2;

% Зона 2
t2 = data.F34_TimeDiff / 100;
T2 = (data.F34_TL + data.F34_TR) / 2;

% Зона 3 (выходная)
t3 = data.F56_TimeDiff / 100;
T3 = (data.F56_TL + data.F56_TR) / 2;

% Лаговая вершина — только предыдущий Tmax
lag_Tmax = Tmax_lag;

% Целевая переменная
T_y = data.Tmax;
fprintf('Tmax: мин=%.0f, макс=%.0f, среднее=%.1f, СКО=%.1f\n', ...
    min(T_y), max(T_y), mean(T_y), std(T_y));

% Векторы данных для каждой вершины
data_zone1 = [t1, T1];
data_zone2 = [t2, T2];
data_zone3 = [t3, T3];
data_lag   = lag_Tmax;

% Матрица признаков для градиентного бустинга (сравнение)
XData_for_gb = [t1, t2, t3, T1, T2, T3, lag_Tmax, data.F_AF, data.F56_P];
YData_for_gb = T_y;

%% 3. Создание графовой модели
% Ядровые функции для трёх зон (физические модели нагрева)
HeatBC_1 = coreFunctions.Heating2DModel(200, 20, 20, 70, 1.5e-5, 0.3, 0.360, 30, 10);
HeatBC_2 = coreFunctions.Heating2DModel(300, 20, 20, 60, 1.0e-5, 0.3, 0.360, 400, 10);
HeatBC_3 = coreFunctions.Heating2DModel(400, 20, 20, 50, 0.5e-5, 0.3, 0.360, 700, 10);

% Лаговая вершина — простая линейная функция (Gamma * lag_Tmax)
LagFunc = coreFunctions.LinearFunction();

% Генераторы начальных alpha и beta
alfaGen = FullRandomAlfaGen(1, 1e1);
betaGen = FullRandomBetaGen(1, 1e2);

% Вершины графа
FirstZone  = Node(1, 30, 'Black', HeatBC_1, 'linear');
SecondZone = Node(2, 30, 'Black', HeatBC_2, 'linear');
ThirdZone  = Node(3, 30, 'White', HeatBC_3, 'linear');
LagNode    = Node(4, 30, 'Black', LagFunc,  'linear');

% Связи между зонами — двусторонние
FirstZone.addEdge(SecondZone);
SecondZone.addEdge(FirstZone);
SecondZone.addEdge(ThirdZone);
ThirdZone.addEdge(SecondZone);

% Связь от лаговой вершины к выходной — односторонняя (4 → 3)
% Лаг не должен зависеть от текущего выхода
LagNode.addEdge(ThirdZone);

% Веса вершин (подбираются под количество рёбер)
NodeWeight = [0.5 0.5 0.5 1];

% Сборка графа
modelShell = GraphShell(betaGen, NodeWeight, ...
    FirstZone, SecondZone, ThirdZone, LagNode);

% Визуализация
modelShell.DrawGraph_New('Модель нагрева с лаговым признаком');

%% 4. Разделение на обучающую и тестовую выборки
% Хронологический сплит (80/20) — сохраняет временной порядок
splitPoint = round(0.8 * totalBatch);
idx_train = (1:splitPoint)';
idx_test  = ((splitPoint+1):totalBatch)';

fprintf('\nОбучающая выборка: %d строк | Тестовая: %d строк\n', ...
    length(idx_train), length(idx_test));

% BWMatrix для графовой модели
XData = repmat(BWMatrix(), totalBatch, 1);
YData = repmat(BWMatrix(), totalBatch, 1);

for i = 1:totalBatch
    XData(i) = XData(i).addRow(data_zone1(i,:));
    XData(i) = XData(i).addRow(data_zone2(i,:));
    XData(i) = XData(i).addRow(data_zone3(i,:));
    XData(i) = XData(i).addRow(data_lag(i,:));
end

for i = 1:totalBatch
    YData(i) = YData(i).addRow(T_y(i,:));
end

XDataTrain = XData(idx_train);
YDataTrain = YData(idx_train);
XDataTest  = XData(idx_test);
YDataTest  = YData(idx_test);

XTrain_GB = XData_for_gb(idx_train, :);
yTrain_GB = YData_for_gb(idx_train);
XTest_GB  = XData_for_gb(idx_test, :);
yTest_GB  = YData_for_gb(idx_test);

%% 5. Настройка параметров обучения
trainerOptions = TrainingOptions( ...
    "LearningRate", 0.01, ...
    "Beta1", 0.6, ...
    "Beta2", 0.8, ...
    "Eps", 1e-8, ...
    "NodeSize", [1 1 1 1], ...
    "Epoches", 500, ...
    "AutoCalibrateClip", true, ...
    "ClipPercentile", 95, ...
    "TargetError", 17, ...
    "Lambda_Agg", 0, ...          % Одна белая вершина
    "Lambda_Alph", 0.1, ...
    "Lambda_Beta", 0.1, ...
    "Lambda_Gamma", 0.1, ...
    "Lambda_Self", 0.1, ...
    "Lambda_Struct", 0.9, ...
    "ErrorMetric", 'mae', ...
    "LossFunction", 'mae', ...
    "TargetNodeIndices", [], ...
    "BatchSize", 1);

%% 6. Обучение графовой модели
fprintf('\n=== Запуск обучения графовой модели ===\n');
trainer = Trainer(modelShell, trainerOptions);
trainer.Train(XDataTrain, YDataTrain, XDataTest, YDataTest);

%% 7. Предсказание на тестовой выборке
numTest = length(idx_test);
act = zeros(1, numTest);
predModel = zeros(1, numTest);

fprintf('\n=== Расчёт предсказаний на тестовой выборке ===\n');
for i = 1:numTest
    act(i) = YDataTest(i).getRow(1);
    result = modelShell.GetCurrentResult(XDataTest(i));
    predModel(i) = result(3);  % Третья вершина — выходная (White)
end

%% 8. Сравнение: градиентный бустинг С лаговым признаком
fprintf('\n=== Градиентный бустинг (с лагом) ===\n');
gbModel_withLag = fitrensemble(XTrain_GB, yTrain_GB, ...
    'Method', 'LSBoost', ...
    'NumLearningCycles', 100, ...
    'Learners', 'tree', ...
    'LearnRate', 0.1);
yPredGB_withLag = predict(gbModel_withLag, XTest_GB);

%% 9. Сравнение: градиентный бустинг БЕЗ лагового признака
fprintf('\n=== Градиентный бустинг (без лага) ===\n');
% Убираем столбец с лаговым Tmax и F_AF+F56_P (для чистоты сравнения)
gbModel_noLag = fitrensemble(XTrain_GB(:, 1:6), yTrain_GB, ...
    'Method', 'LSBoost', ...
    'NumLearningCycles', 100, ...
    'Learners', 'tree', ...
    'LearnRate', 0.1);
yPredGB_noLag = predict(gbModel_noLag, XTest_GB(:, 1:6));

%% 10. Диагностика
diagnostics_BW = plotHeatingPrediction(act, predModel, ...
    (splitPoint+1):totalBatch, 'BW Graph (с лагом)');
diagnostics_GB_withLag = plotHeatingPrediction(yTest_GB, yPredGB_withLag, ...
    (splitPoint+1):totalBatch, 'GBoost (с лагом)');
diagnostics_GB_noLag = plotHeatingPrediction(yTest_GB, yPredGB_noLag, ...
    (splitPoint+1):totalBatch, 'GBoost (без лага)');

%% 11. Сводка результатов
fprintf('\n╔══════════════════════════════════════════════╗\n');
fprintf('║         СВОДКА РЕЗУЛЬТАТОВ                   ║\n');
fprintf('╠══════════════════════════════════════════════╣\n');
fprintf('║ Модель              │ MAE     │ R²          ║\n');
fprintf('╠══════════════════════════════════════════════╣\n');
fprintf('║ BW Graph (с лагом)   │ %6.1f°C │ %+.4f      ║\n', ...
    diagnostics_BW.MAE, diagnostics_BW.R2);
fprintf('║ GBoost (с лагом)     │ %6.1f°C │ %+.4f      ║\n', ...
    diagnostics_GB_withLag.MAE, diagnostics_GB_withLag.R2);
fprintf('║ GBoost (без лага)    │ %6.1f°C │ %+.4f      ║\n', ...
    diagnostics_GB_noLag.MAE, diagnostics_GB_noLag.R2);
fprintf('╚══════════════════════════════════════════════╝\n');

%% 12. Демонстрационный прогноз
fprintf('\n=== Демонстрационный прогноз ===\n');
ValidMatrix = BWMatrix();
ValidMatrix = ValidMatrix.addRow([100, 1000]);     % Зона 1: 100с, 1000°C
ValidMatrix = ValidMatrix.addRow([60, 1220]);      % Зона 2: 60с, 1220°C
ValidMatrix = ValidMatrix.addRow([50, 1240]);      % Зона 3: 50с, 1240°C
ValidMatrix = ValidMatrix.addRow([1020]);          % Лаг: Tmax_prev = 1020°C
result = modelShell.GetCurrentResult(ValidMatrix);
prediction = result(3);
fprintf('При Tmax_prev=1020°C → прогноз Tmax = %.1f°C\n', prediction);

%% ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

function diagnostics = plotHeatingPrediction(y_true, y_pred, time_vector, model_name)
% Диагностика модели прогнозирования нагрева
    if nargin < 3 || isempty(time_vector)
        time_vector = 1:length(y_true);
    end
    if nargin < 4
        model_name = 'Модель';
    end

    y_true = y_true(:);
    y_pred = y_pred(:);
    time_vector = time_vector(:);

    figure('Position', [100, 100, 1400, 900]);

    % 1. Scatter: Предсказания vs Реальность
    subplot(2, 3, 1);
    plot(y_true, y_pred, 'b.', 'MarkerSize', 8);
    hold on;
    min_val = min([y_true; y_pred]);
    max_val = max([y_true; y_pred]);
    plot([min_val, max_val], [min_val, max_val], 'r-', 'LineWidth', 2);
    coeffs = polyfit(y_true, y_pred, 1);
    y_fit = polyval(coeffs, [min_val, max_val]);
    plot([min_val, max_val], y_fit, 'g--', 'LineWidth', 1.5);
    xlabel('Реальная температура (°C)');
    ylabel('Предсказанная температура (°C)');
    title(sprintf('%s: Scatter', model_name));
    legend('Предсказания', 'Идеал', 'Тренд', 'Location', 'best');
    grid on;

    % 2. Временной ряд
    subplot(2, 3, 2);
    plot(time_vector, y_true, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Реальная');
    hold on;
    plot(time_vector, y_pred, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Предсказанная');
    xlabel('Индекс наблюдения');
    ylabel('Температура (°C)');
    title('Временной ряд');
    legend('Location', 'best');
    grid on;

    % 3. Ошибки во времени
    subplot(2, 3, 3);
    errors = y_pred - y_true;
    plot(time_vector, errors, 'b-', 'LineWidth', 1);
    hold on;
    plot(time_vector, zeros(size(time_vector)), 'r--', 'LineWidth', 1);
    std_err = std(errors);
    plot(time_vector, 2*std_err * ones(size(time_vector)), 'g--', 'LineWidth', 0.5);
    plot(time_vector, -2*std_err * ones(size(time_vector)), 'g--', 'LineWidth', 0.5);
    xlabel('Индекс наблюдения');
    ylabel('Ошибка (°C)');
    title(sprintf('Ошибки (MAE = %.1f°C)', mean(abs(errors))));
    legend('Ошибка', 'Ноль', '\pm2\sigma', 'Location', 'best');
    grid on;

    % 4. Гистограмма ошибок
    subplot(2, 3, 4);
    histogram(errors, 30, 'Normalization', 'pdf', 'FaceColor', [0.8 0.8 1]);
    hold on;
    x_range = linspace(min(errors), max(errors), 100);
    y_norm = normpdf(x_range, mean(errors), std(errors));
    plot(x_range, y_norm, 'r-', 'LineWidth', 2);
    xlabel('Ошибка (°C)');
    ylabel('Плотность');
    title('Распределение ошибок');
    legend('Ошибки', 'Нормальное', 'Location', 'best');
    grid on;

    % 5. Автокорреляция ошибок
    subplot(2, 3, 5);
    [acf, lags] = xcorr(errors - mean(errors), 20, 'normalized');
    lags = lags(21:end);
    acf = acf(21:end);
    stem(lags, acf, 'filled', 'LineWidth', 1.5);
    hold on;
    conf_level = 1.96 / sqrt(length(errors));
    plot([0, 20], [conf_level, conf_level], 'r--');
    plot([0, 20], [-conf_level, -conf_level], 'r--');
    xlabel('Лаг');
    ylabel('Автокорреляция');
    title('Автокорреляция ошибок');
    grid on;
    ylim([-1, 1]);

    % 6. Остатки vs Предсказания
    subplot(2, 3, 6);
    plot(y_pred, errors, 'b.', 'MarkerSize', 8);
    hold on;
    plot([min(y_pred), max(y_pred)], [0, 0], 'r--', 'LineWidth', 1);
    xlabel('Предсказанная температура (°C)');
    ylabel('Остатки (°C)');
    title('Остатки vs Предсказания');
    grid on;

    sgtitle(sprintf('Диагностика: %s', model_name), 'FontSize', 14, 'FontWeight', 'bold');

    % Расчёт метрик
    diagnostics = calcMetrics(y_true, y_pred);

    fprintf('\n=== %s ===\n', model_name);
    fprintf('R² = %.4f | MAE = %.1f°C | RMSE = %.1f°C\n', ...
        diagnostics.R2, diagnostics.MAE, diagnostics.RMSE);
    fprintf('Медианная ошибка: %.1f°C | Макс. ошибка: %.1f°C\n', ...
        diagnostics.MedAE, diagnostics.max_error);
end

function metrics = calcMetrics(y_true, y_pred)
    errors = y_pred - y_true;
    abs_errors = abs(errors);
    ss_res = sum(errors.^2);
    ss_tot = sum((y_true - mean(y_true)).^2);
    metrics.R2 = 1 - ss_res/ss_tot;
    metrics.MAE = mean(abs_errors);
    metrics.RMSE = sqrt(mean(errors.^2));
    metrics.MedAE = median(abs_errors);
    metrics.max_error = max(abs_errors);
    metrics.y_mean = mean(y_true);
    metrics.y_std = std(y_true);
end
