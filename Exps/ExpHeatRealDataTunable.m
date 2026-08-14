
%% Fast Start — ExpHeatRealData с настраиваемыми ядровыми функциями
% Использует Heating2DTunableModel вместо Heating2DModel
% Параметры h и alpha настраиваются автоматически через Trainer.Compute_V5
clear; clc; close all;
rng(1111);

import BWGraph.*;
import BWGraph.CustomMatrix.*;
import BWGraph.RandomGenerator.*;
import BWGraph.Trainer.*;

% Ядровые функции — Heating2DTunableModel (h и alpha обучаемые)
% Параметры: h, nx, ny, lambda, alpha, Lx, Ly, T0, nt, learningRate
HeatBC_1 = coreFunctions.Heating2DTunableModel(200, 20, 20, 70, 1.5e-5, 0.3, 0.360, 30, 10, 0.01);
HeatBC_2 = coreFunctions.Heating2DTunableModel(300, 20, 20, 60, 1.0e-5, 0.3, 0.360, 400, 10, 0.01);
HeatBC_3 = coreFunctions.Heating2DTunableModel(400, 20, 20, 50, 0.5e-5, 0.3, 0.360, 700, 10, 0.01);

alfaGen = FullRandomAlfaGen(1, 1e1);
betaGen = FullRandomBetaGen(1, 1e2);

FirstZone = Node(1, 30, 'Black', HeatBC_1, 'linear');
SecondZone = Node(2, 30, 'Black', HeatBC_2, 'linear');
ThirdZone = Node(3, 30, 'White', HeatBC_3, 'linear');

FirstZone.addEdge(SecondZone);
SecondZone.addEdge(ThirdZone);

NodeWeight = [0.5 0.5 1];

modelShell = GraphShell(betaGen, NodeWeight, FirstZone, SecondZone, ThirdZone);

if ~exist("data", 'var')
    data = readtable("\Новые данные\06.05.2026.xlsx");
end

%% Подготовка данных
totalBatch = 100;
initial = 1;

t1 = table2array(data(initial:totalBatch, "t12_H").*3600) + table2array(data(initial:totalBatch, "t12_M").*60);
t2 = table2array(data(initial:totalBatch, "t34_H").*3600) + table2array(data(initial:totalBatch, "t34_M").*60);
t3 = table2array(data(initial:totalBatch, "t56_H").*3600) + table2array(data(initial:totalBatch, "t56_M").*60);

T1 = data{initial:totalBatch, "T12_Avg"};
T2 = data{initial:totalBatch, "T34_Avg"};
T3 = data{initial:totalBatch, "T56_Avg"};
T_y = data{initial:totalBatch, "T_Res_Max"};

fprintf('Новый размер выборки: %d\n', totalBatch);

data_for_one = [t1, T1];
data_for_two = [t2, T2];
data_for_three = [t3, T3];

XData_for_gb = [t1, t2, t3, T1, T2, T3];
YData_for_gb = T_y;

XData = repmat(BWMatrix(), (totalBatch - initial) + 1, 1);
YData = repmat(BWMatrix(), (totalBatch - initial) + 1, 1);

for i = 1:(totalBatch - initial) + 1
    XData(i) = XData(i).addRow(data_for_one(i, :));
    XData(i) = XData(i).addRow(data_for_two(i, :));
    XData(i) = XData(i).addRow(data_for_three(i, :));
end

for i = 1:(totalBatch - initial) + 1
    YData(i) = YData(i).addRow(T_y(i, :));
end

indices = randperm((totalBatch - initial) + 1);
splitPoint = round(0.7 * (totalBatch - initial) + 1);
idx_train = indices(1:splitPoint);
idx_test = indices(splitPoint + 1:end);

XDataTrain = XData(idx_train);
YDataTrain = YData(idx_train);
XDataTest = XData(idx_test);
YDataTest = YData(idx_test);

XTrain_GB = XData_for_gb(idx_train, :);
yTrain_GB = YData_for_gb(idx_train);
XTest_GB = XData_for_gb(idx_test, :);
yTest_GB = YData_for_gb(idx_test);

%% Отрисовать граф
modelShell.DrawGraph_New('Модель нагрева (настраиваемые h, alpha)');

%% Настройка учителя
trainerOptions = TrainingOptions( ...
    "LearningRate", 0.01, ...
    "Beta1", 0.6, ...
    "Beta2", 0.8, ...
    "Eps", 1e-8, ...
    "NodeSize", [1 1 1], ...
    "Epoches", 500, ...
    "ClipUp", 1e18, ...
    "ClipDown", -1e18, ...
    "TargetError", 10, ...
    "Lambda_Agg", 0, ...
    "Lambda_Alph", 0.1, ...
    "Lambda_Beta", 0.1, ...
    "Lambda_Gamma", 0.1, ...
    "Lambda_Self", 0.1, ...
    "Lambda_Struct", 0.9, ...
    "ErrorMetric", 'mae', ...
    "LossFunction", 'mae', ...
    "TargetNodeIndices", [], ...
    "BatchSize", 1);

trainer = Trainer(modelShell, trainerOptions);

%% Вывод начальных параметров ядровых функций
fprintf('\n========== Начальные параметры Heating2DTunableModel ==========\n');
params1 = HeatBC_1.GetTunableParameters();
params2 = HeatBC_2.GetTunableParameters();
params3 = HeatBC_3.GetTunableParameters();
fprintf('HeatBC_1: h=%.4f, alpha=%.2e\n', params1.h, params1.alpha);
fprintf('HeatBC_2: h=%.4f, alpha=%.2e\n', params2.h, params2.alpha);
fprintf('HeatBC_3: h=%.4f, alpha=%.2e\n', params3.h, params3.alpha);
fprintf('===============================================================\n\n');

%% Запуск процесса обучения
trainer.Train(XDataTrain, YDataTrain, XDataTest, YDataTest);

%% Вывод финальных параметров ядровых функций
fprintf('\n========== Финальные параметры Heating2DTunableModel ==========\n');
params1 = HeatBC_1.GetTunableParameters();
params2 = HeatBC_2.GetTunableParameters();
params3 = HeatBC_3.GetTunableParameters();
fprintf('HeatBC_1: h=%.4f, alpha=%.2e\n', params1.h, params1.alpha);
fprintf('HeatBC_2: h=%.4f, alpha=%.2e\n', params2.h, params2.alpha);
fprintf('HeatBC_3: h=%.4f, alpha=%.2e\n', params3.h, params3.alpha);
fprintf('===============================================================\n\n');

%% Предсказание на тестовой выборке
numTestSamples = size(XDataTest, 1);

for i = 1:numTestSamples
    act(i) = YDataTest(i).getRow(1);
    result = modelShell.GetCurrentResult(XDataTest(i));
    predModel(i) = result(3);
end

%% Чек модели
diagnostics_BW = plotHeatingPrediction(act, predModel);

%% Градиентный бустинг для сравнения
gbModel = fitrensemble(XTrain_GB, yTrain_GB, ...
    'Method', 'LSBoost', ...
    'NumLearningCycles', 100, ...
    'Learners', 'tree', ...
    'LearnRate', 0.1);

yPredGB = predict(gbModel, XTest_GB);

mseGB = mean((yPredGB - yTest_GB).^2);
r2GB = 1 - sum((yTest_GB - yPredGB).^2) / sum((yTest_GB - mean(yTest_GB)).^2);
fprintf('Градиентный бустинг - MSE: %.4f, R2: %.4f\n', mseGB, r2GB);

scatter(yTest_GB, yPredGB);
hold on;
plot([min(yTest_GB) max(yTest_GB)], [min(yTest_GB) max(yTest_GB)], 'r--');
xlabel('Истинные значения');
ylabel('Предсказания GB');
title('Градиентный бустинг');
grid on;

%% Чек LSBoost
diagnostics_LSBoost = plotHeatingPrediction(act, yPredGB);

%% ===== Вспомогательные функции =====

function diagnostics = plotHeatingPrediction(y_true, y_pred, time_vector, model_name)
    if nargin < 3
        time_vector = 1:length(y_true);
    end
    if nargin < 4
        model_name = 'Модель';
    end

    y_true = y_true(:);
    y_pred = y_pred(:);
    time_vector = time_vector(:);

    assert(length(y_true) == length(y_pred), 'Длины y_true и y_pred должны совпадать');
    assert(length(y_true) == length(time_vector), 'Длины y_true и time_vector должны совпадать');

    figure('Position', [100, 100, 1400, 900]);

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
    title(sprintf('%s: Scatter Plot\nКрасный - идеал, Зеленый - факт', model_name));
    legend('Предсказания', 'Идеал', 'Тренд', 'Location', 'best');
    grid on;
    axis equal;
    text(min_val + 0.05*(max_val-min_val), max_val - 0.1*(max_val-min_val), ...
         sprintf('Наклон тренда: %.2f', coeffs(1)), 'FontSize', 10);

    subplot(2, 3, 2);
    plot(time_vector, y_true, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Реальная');
    hold on;
    plot(time_vector, y_pred, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Предсказанная');
    y_mean = mean(y_true) * ones(size(time_vector));
    plot(time_vector, y_mean, 'g-', 'LineWidth', 1, 'DisplayName', 'Средняя');
    xlabel('Время/Индекс');
    ylabel('Температура (°C)');
    title('Временной ряд: Реальность vs Предсказания');
    legend('Location', 'best');
    grid on;

    subplot(2, 3, 3);
    errors = y_pred - y_true;
    plot(time_vector, errors, 'b-', 'LineWidth', 1);
    hold on;
    plot(time_vector, zeros(size(time_vector)), 'r--', 'LineWidth', 1);
    std_err = std(errors);
    plot(time_vector, 2*std_err * ones(size(time_vector)), 'g--', 'LineWidth', 0.5);
    plot(time_vector, -2*std_err * ones(size(time_vector)), 'g--', 'LineWidth', 0.5);
    xlabel('Время/Индекс');
    ylabel('Ошибка (°C)');
    title(sprintf('Ошибка предсказания (MAE = %.2f°C)', mean(abs(errors))));
    legend('Ошибка', 'Ноль', '±2σ', 'Location', 'best');
    grid on;

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

    subplot(2, 3, 6);
    plot(y_pred, errors, 'b.', 'MarkerSize', 8);
    hold on;
    plot([min(y_pred), max(y_pred)], [0, 0], 'r--', 'LineWidth', 1);
    window_size = max(3, floor(length(errors)/20));
    [y_pred_sorted, sort_idx] = sort(y_pred);
    errors_sorted = errors(sort_idx);
    moving_std = movstd(errors_sorted, window_size);
    moving_mean = movmean(errors_sorted, window_size);
    plot(y_pred_sorted, moving_mean, 'g-', 'LineWidth', 2);
    plot(y_pred_sorted, moving_std, 'm-', 'LineWidth', 1);
    plot(y_pred_sorted, -moving_std, 'm-', 'LineWidth', 1);
    xlabel('Предсказанная температура (°C)');
    ylabel('Остатки (°C)');
    title('Остатки vs Предсказания');
    legend('Остатки', 'Ноль', 'Скользящее среднее', '± скользящее Std', 'Location', 'best');
    grid on;

    sgtitle(sprintf('Диагностика модели: %s', model_name), 'FontSize', 14, 'FontWeight', 'bold');

    diagnostics = calculateMetrics(y_true, y_pred);

    fprintf('\n========== МЕТРИКИ МОДЕЛИ: %s ==========\n', model_name);
    fprintf('R2 (коэффициент детерминации): %.4f\n', diagnostics.R2);
    fprintf('MAE (средняя абсолютная ошибка): %.2f °C\n', diagnostics.MAE);
    fprintf('RMSE (среднеквадратичная ошибка): %.2f °C\n', diagnostics.RMSE);
    fprintf('MAPE (средняя относительная ошибка): %.2f %%\n', diagnostics.MAPE);
    fprintf('Медианная абсолютная ошибка: %.2f °C\n', diagnostics.MedAE);
    fprintf('\nСтатистика целевой переменной:\n');
    fprintf('  Среднее: %.2f °C\n', diagnostics.y_mean);
    fprintf('  Std: %.2f °C\n', diagnostics.y_std);
    fprintf('  Min: %.2f °C\n', diagnostics.y_min);
    fprintf('  Max: %.2f °C\n', diagnostics.y_max);
    fprintf('  Размах: %.2f °C\n', diagnostics.y_range);
    fprintf('\nОтношение RMSE к Std: %.2f (должно быть < 1 для полезной модели)\n', ...
            diagnostics.RMSE / diagnostics.y_std);
    fprintf('========================================\n\n');

    printDiagnostics(diagnostics);
end

function metrics = calculateMetrics(y_true, y_pred)
    n = length(y_true);
    errors = y_pred - y_true;
    abs_errors = abs(errors);

    ss_res = sum(errors.^2);
    ss_tot = sum((y_true - mean(y_true)).^2);
    metrics.R2 = 1 - ss_res/ss_tot;

    metrics.MAE = mean(abs_errors);
    metrics.RMSE = sqrt(mean(errors.^2));

    non_zero_idx = y_true ~= 0;
    if any(non_zero_idx)
        metrics.MAPE = mean(abs_errors(non_zero_idx) ./ abs(y_true(non_zero_idx))) * 100;
    else
        metrics.MAPE = NaN;
    end

    metrics.MedAE = median(abs_errors);
    metrics.y_mean = mean(y_true);
    metrics.y_std = std(y_true);
    metrics.y_min = min(y_true);
    metrics.y_max = max(y_true);
    metrics.y_range = metrics.y_max - metrics.y_min;
    metrics.max_error = max(abs_errors);
    metrics.error_std = std(errors);
end

function printDiagnostics(diagnostics)
    fprintf('========== ДИАГНОСТИЧЕСКИЕ ВЫВОДЫ ==========\n');

    if diagnostics.R2 < 0
        fprintf('ПРОБЛЕМА: R2 = %.2f (отрицательный!)\n', diagnostics.R2);
        fprintf('   Модель работает ХУЖЕ, чем просто предсказание среднего.\n');
    elseif diagnostics.R2 < 0.3
        fprintf('R2 = %.2f (низкий)\n', diagnostics.R2);
        fprintf('   Модель объясняет только %.0f%% вариации данных.\n', diagnostics.R2*100);
    elseif diagnostics.R2 < 0.7
        fprintf('R2 = %.2f (средний)\n', diagnostics.R2);
        fprintf('   Модель объясняет %.0f%% вариации данных.\n', diagnostics.R2*100);
    else
        fprintf('R2 = %.2f (отличный!)\n', diagnostics.R2);
    end

    rmse_std_ratio = diagnostics.RMSE / diagnostics.y_std;
    if rmse_std_ratio > 1
        fprintf('RMSE (%.2f) БОЛЬШЕ стандартного отклонения (%.2f)\n', ...
                diagnostics.RMSE, diagnostics.y_std);
    elseif rmse_std_ratio > 0.7
        fprintf('RMSE составляет %.0f%% от стандартного отклонения\n', rmse_std_ratio*100);
    else
        fprintf('RMSE составляет %.0f%% от стандартного отклонения\n', rmse_std_ratio*100);
    end

    if diagnostics.MAE > 0.2 * diagnostics.y_range
        fprintf('MAE (%.2f) составляет >20%% от размаха данных (%.2f)\n', ...
                diagnostics.MAE, diagnostics.y_range);
    end

    fprintf('\nРЕКОМЕНДАЦИИ:\n');
    if diagnostics.R2 < 0
        fprintf('   - Проверьте, нет ли перепутанных меток (переменных)\n');
        fprintf('   - Проверьте выбросы в данных\n');
    elseif diagnostics.R2 < 0.3
        fprintf('   - Добавьте больше признаков (историю нагрева)\n');
        fprintf('   - Проверьте лаги\n');
        fprintf('   - Попробуйте другую модель\n');
    end

    fprintf('==============================================\n');
end