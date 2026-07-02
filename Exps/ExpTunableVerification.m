%% Эксперимент: сравнение Heating2DModel vs Heating2DTunableModel
% Доказательство эффективности настройки физических параметров
% Стратегия: синтетические данные с ИЗВЕСТНЫМИ истинными параметрами,
% обучение с НАМЕРЕННО ИСКАЖЁННЫМИ начальными значениями
clear; clc; close all;
rng(22);

import BWGraph.*;
import BWGraph.CustomMatrix.*;
import BWGraph.RandomGenerator.*;
import BWGraph.Trainer.*;

%% ===== Часть 1: Истинные и искажённые параметры =====

% Истинные параметры (используются для генерации данных)
h_true    = [200, 300, 400];       % Коэффициент теплоотдачи
alpha_true = [1.5e-5, 1.0e-5, 0.5e-5]; % Температуропроводность

% Начальные (искажённые) параметры для обучения
h_init    = [100, 500, 200];       % Намеренно отличаются от истинных
alpha_init = [3.0e-5, 0.5e-5, 2.0e-5];

fprintf('========== Параметры эксперимента ==========\n');
fprintf('Истинные:      h = [%.0f, %.0f, %.0f], alpha = [%.1e, %.1e, %.1e]\n', ...
    h_true, alpha_true);
fprintf('Искажённые:    h = [%.0f, %.0f, %.0f], alpha = [%.1e, %.1e, %.1e]\n', ...
    h_init, alpha_init);
fprintf('=============================================\n\n');

%% ===== Часть 2: Генерация синтетических данных (с истинными параметрами) =====

% Ядровые функции с ИСТИННЫМИ параметрами — только для генерации Y
HeatTrue_1 = coreFunctions.Heating2DModel(h_true(1), 20, 20, 70, alpha_true(1), 0.3, 0.360, 30, 10);
HeatTrue_2 = coreFunctions.Heating2DModel(h_true(2), 20, 20, 70, alpha_true(2), 0.3, 0.360, 30, 10);
HeatTrue_3 = coreFunctions.Heating2DModel(h_true(3), 20, 20, 70, alpha_true(3), 0.3, 0.360, 30, 10);

numSamples = 50;

% Диапазоны входных данных
timeRange  = {[1000, 11000], [1000, 15000], [1000, 9000]};
TinfRange  = {[900, 1200],   [800, 1150],   [900, 1100]};

timeVals = cell(1,3); TinfVals = cell(1,3);
for v = 1:3
    timeVals{v} = rand(numSamples, 1) * (timeRange{v}(2) - timeRange{v}(1)) + timeRange{v}(1);
    TinfVals{v} = rand(numSamples, 1) * (TinfRange{v}(2) - TinfRange{v}(1)) + TinfRange{v}(1);
end

% Генерация Y через истинные параметры (композиция трёх нагревов)
trueFunctions = {HeatTrue_1, HeatTrue_2, HeatTrue_3};
Y_target = zeros(numSamples, 1);
for i = 1:numSamples
    c1 = trueFunctions{1}.CalcCoreFunction([timeVals{1}(i); TinfVals{1}(i)]);
    c2 = trueFunctions{2}.CalcCoreFunction([timeVals{2}(i); TinfVals{2}(i)]);
    c3 = trueFunctions{3}.CalcCoreFunction([timeVals{3}(i); TinfVals{3}(i)]);
    Y_target(i) = c1 + 0.1 * c2 + 0.15 * c3;
end

% Формирование XData / YData для графа
XData = repmat(BWMatrix(), numSamples, 1);
YData = repmat(BWMatrix(), numSamples, 1);
for i = 1:numSamples
    XData(i) = XData(i).addRow([timeVals{1}(i); TinfVals{1}(i)]);
    XData(i) = XData(i).addRow([timeVals{2}(i); TinfVals{2}(i)]);
    XData(i) = XData(i).addRow([timeVals{3}(i); TinfVals{3}(i)]);
    YData(i) = YData(i).addRow(Y_target(i));
end

% Train/test split
cv = cvpartition(numSamples, 'HoldOut', 0.3);
idxTrain = training(cv); idxTest = test(cv);
XDataTrain = XData(idxTrain); YDataTrain = YData(idxTrain);
XDataTest  = XData(idxTest);  YDataTest  = YData(idxTest);

% Данные для градиентного бустинга
XDataGB = [timeVals{1}, timeVals{2}, timeVals{3}, TinfVals{1}, TinfVals{2}, TinfVals{3}];
XTrainGB = XDataGB(idxTrain, :); yTrainGB = Y_target(idxTrain);
XTestGB  = XDataGB(idxTest, :);  yTestGB  = Y_target(idxTest);

fprintf('Обучающих примеров: %d, тестовых: %d\n\n', sum(idxTrain), sum(idxTest));

%% ===== Часть 3: Эксперимент A — Baseline (фиксированные параметры) =====

fprintf('========== ЭКСПЕРИМЕНТ A: Heating2DModel (baseline) ==========\n');

HeatFix_1 = coreFunctions.Heating2DModel(h_init(1), 20, 20, 70, alpha_init(1), 0.3, 0.360, 30, 10);
HeatFix_2 = coreFunctions.Heating2DModel(h_init(2), 20, 20, 70, alpha_init(2), 0.3, 0.360, 30, 10);
HeatFix_3 = coreFunctions.Heating2DModel(h_init(3), 20, 20, 70, alpha_init(3), 0.3, 0.360, 30, 10);

alfaGen = FullRandomAlfaGen(1, 5);
betaGen = FullRandomBetaGen(1, 1e2);

nodeA_fix = Node(1, 1, 'White', HeatFix_1, 'linear');
nodeB_fix = Node(2, 1, 'Black', HeatFix_2, 'linear');
nodeC_fix = Node(3, 1, 'Black', HeatFix_3, 'linear');
nodeB_fix.addEdge(nodeA_fix);
nodeC_fix.addEdge(nodeA_fix);

modelShell_fix = GraphShell(betaGen, [1 0.5 0.5], nodeA_fix, nodeB_fix, nodeC_fix);

trainerOpts = TrainingOptions(...
    "LearningRate", 0.01, "Beta1", 0.9, "Beta2", 0.999, "Eps", 1e-8, ...
    "NodeSize", [1, 1, 1], "Epoches", 300, ...
    "ClipUp", 1e7, "ClipDown", -1e7, "TargetError", 10, ...
    "Lambda_Agg", 0.0, "Lambda_Alph", 0.4, "Lambda_Beta", 0.4, ...
    "Lambda_Self", 0.1, "Lambda_Struct", 0.9, "Lambda_Gamma", 0.4, ...
    "ErrorMetric", 'mae', "LossFunction", 'mse', ...
    "TargetNodeIndices", [], "BatchSize", 1);

trainer_fix = Trainer(modelShell_fix, trainerOpts);

% История ошибок (перехватываем через публичное свойство, если доступно)
tic;
trainer_fix.Train(XDataTrain, YDataTrain, XDataTest, YDataTest);
time_fix = toc;
%%
% Предсказания
pred_fix = zeros(sum(idxTest), 1);
actual   = zeros(sum(idxTest), 1);
for j = 1:sum(idxTest)
    result = modelShell_fix.GetCurrentResult(XDataTest(j));
    pred_fix(j) = result(1);
    actual(j) = YDataTest(j).getRow(1);
end

mae_fix  = mean(abs(pred_fix - actual));
rmse_fix = sqrt(mean((pred_fix - actual).^2));
r2_fix   = 1 - sum((actual - pred_fix).^2) / sum((actual - mean(actual)).^2);

fprintf('Baseline — MAE: %.4f, RMSE: %.4f, R2: %.4f, Время: %.1f с\n', ...
    mae_fix, rmse_fix, r2_fix, time_fix);

% Сохраняем для сравнения
results.fix.mae  = mae_fix;
results.fix.rmse = rmse_fix;
results.fix.r2   = r2_fix;
results.fix.pred = pred_fix;
results.fix.time = time_fix;
results.fix.h_final    = [h_init(1), h_init(2), h_init(3)];
results.fix.alpha_final = [alpha_init(1), alpha_init(2), alpha_init(3)];

%% ===== Часть 4: Эксперимент B — Heating2DTunableModel (настраиваемые) =====

fprintf('\n========== ЭКСПЕРИМЕНТ B: Heating2DTunableModel ==========\n');

HeatTune_1 = coreFunctions.Heating2DTunableModel(h_init(1), 20, 20, 70, alpha_init(1), 0.3, 0.360, 30, 10, 0.1);
HeatTune_2 = coreFunctions.Heating2DTunableModel(h_init(2), 20, 20, 60, alpha_init(2), 0.3, 0.360, 400, 10, 0.1);
HeatTune_3 = coreFunctions.Heating2DTunableModel(h_init(3), 20, 20, 50, alpha_init(3), 0.3, 0.360, 700, 10, 0.1);
alfaGen = FullRandomAlfaGen(1, 5);
betaGen = FullRandomBetaGen(1, 1e2);
nodeA_tune = Node(1, 1, 'White', HeatTune_1, 'linear');
nodeB_tune = Node(2, 1, 'Black', HeatTune_2, 'linear');
nodeC_tune = Node(3, 1, 'Black', HeatTune_3, 'linear');
nodeB_tune.addEdge(nodeA_tune);
nodeC_tune.addEdge(nodeA_tune);
trainerOpts = TrainingOptions(...
    "LearningRate", 0.01, "Beta1", 0.9, "Beta2", 0.999, "Eps", 1e-8, ...
    "NodeSize", [1, 1, 1], "Epoches", 300, ...
    "ClipUp", 1e7, "ClipDown", -1e7, "TargetError", 10, ...
    "Lambda_Agg", 0.0, "Lambda_Alph", 0.4, "Lambda_Beta", 0.4, ...
    "Lambda_Self", 0.1, "Lambda_Struct", 0.9, "Lambda_Gamma", 0.4, ...
    "ErrorMetric", 'mae', "LossFunction", 'mse', ...
    "TargetNodeIndices", [], "BatchSize", 1);

modelShell_tune = GraphShell(betaGen, [1 0.5 0.5], nodeA_tune, nodeB_tune, nodeC_tune);

trainer_tune = Trainer(modelShell_tune, trainerOpts);
tic;
trainer_tune.Train(XDataTrain, YDataTrain, XDataTest, YDataTest);
time_tune = toc;
%%
% Снимаем финальные параметры
params1 = HeatTune_1.GetTunableParameters();
params2 = HeatTune_2.GetTunableParameters();
params3 = HeatTune_3.GetTunableParameters();

h_final_tune     = [params1.h, params2.h, params3.h];
alpha_final_tune = [params1.alpha, params2.alpha, params3.alpha];

% Предсказания
pred_tune = zeros(sum(idxTest), 1);
for j = 1:sum(idxTest)
    result = modelShell_tune.GetCurrentResult(XDataTest(j));
    pred_tune(j) = result(1);
end

mae_tune  = mean(abs(pred_tune - actual));
rmse_tune = sqrt(mean((pred_tune - actual).^2));
r2_tune   = 1 - sum((actual - pred_tune).^2) / sum((actual - mean(actual)).^2);

fprintf('Tunable  — MAE: %.4f, RMSE: %.4f, R2: %.4f, Время: %.1f с\n', ...
    mae_tune, rmse_tune, r2_tune, time_tune);

results.tune.mae  = mae_tune;
results.tune.rmse = rmse_tune;
results.tune.r2   = r2_tune;
results.tune.pred = pred_tune;
results.tune.time = time_tune;
results.tune.h_final    = h_final_tune;
results.tune.alpha_final = alpha_final_tune;

%% ===== Часть 5: Сравнительный анализ =====

fprintf('\n========== СРАВНИТЕЛЬНЫЙ АНАЛИЗ ==========\n');
fprintf('Метрика              | Baseline (Fix) | Tunable        | Улучшение\n');
fprintf('-------------------- | -------------- | -------------- | ---------\n');
fprintf('MAE                  | %.4f          | %.4f          | %.1f%%\n', ...
    mae_fix, mae_tune, (1 - mae_tune/mae_fix) * 100);
fprintf('RMSE                 | %.4f          | %.4f          | %.1f%%\n', ...
    rmse_fix, rmse_tune, (1 - rmse_tune/rmse_fix) * 100);
fprintf('R2                   | %.4f          | %.4f          | %+.4f\n', ...
    r2_fix, r2_tune, r2_tune - r2_fix);
fprintf('Время обучения (с)   | %.1f           | %.1f           | %.1fx\n', ...
    time_fix, time_tune, time_tune / time_fix);

%% ===== Часть 6: Дрейф параметров =====

fprintf('\n========== ДРЕЙФ ПАРАМЕТРОВ ==========\n');
fprintf('Параметр  | Истинное | Начальное | Финальное  | Ошибка нач. | Ошибка кон.\n');
fprintf('--------- | -------- | --------- | ---------- | ----------- | ----------\n');
for v = 1:3
    fprintf('h_%d       | %8.0f | %9.0f | %10.1f | %10.0f | %10.1f\n', ...
        v, h_true(v), h_init(v), h_final_tune(v), ...
        abs(h_init(v) - h_true(v)), abs(h_final_tune(v) - h_true(v)));
    fprintf('alpha_%d   | %8.1e | %9.1e | %10.1e | %10.1e | %10.1e\n', ...
        v, alpha_true(v), alpha_init(v), alpha_final_tune(v), ...
        abs(alpha_init(v) - alpha_true(v)), abs(alpha_final_tune(v) - alpha_true(v)));
end

% Относительное улучшение параметров
for v = 1:3
    err_init_h = abs(h_init(v) - h_true(v));
    err_final_h = abs(h_final_tune(v) - h_true(v));
    err_init_a = abs(alpha_init(v) - alpha_true(v));
    err_final_a = abs(alpha_final_tune(v) - alpha_true(v));
    if err_init_h > 0
        fprintf('h_%d:      ошибка уменьшилась на %.1f%%\n', v, (1 - err_final_h/err_init_h)*100);
    end
    if err_init_a > 0
        fprintf('alpha_%d:  ошибка уменьшилась на %.1f%%\n', v, (1 - err_final_a/err_init_a)*100);
    end
end

%% ===== Часть 7: Статистическая значимость =====

fprintf('\n========== СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ ==========\n');

residuals_fix  = actual - pred_fix;
residuals_tune = actual - pred_tune;

% Парный t-тест на абсолютных ошибках
abs_err_fix  = abs(residuals_fix);
abs_err_tune = abs(residuals_tune);
[h_ttest, p_ttest] = ttest(abs_err_fix, abs_err_tune);
fprintf('Парный t-тест (|ошибка|): p = %.6f, H0 отвергнута: %d\n', p_ttest, h_ttest);

% Wilcoxon signed-rank (непараметрический)
[p_wilcoxon, h_wilcoxon] = signrank(abs_err_fix, abs_err_tune);
fprintf('Wilcoxon signed-rank:      p = %.6f, H0 отвергнута: %d\n', p_wilcoxon, h_wilcoxon);

% Дисперсионный анализ
fprintf('D[остатки fix]:  %.4f\n', var(residuals_fix));
fprintf('D[остатки tune]: %.4f\n', var(residuals_tune));

% Тест на равенство дисперсий
[h_var, p_var] = vartest2(residuals_fix, residuals_tune);
fprintf('F-тест дисперсий: p = %.4f (H0: дисперсии равны)\n', p_var);

%% ===== Часть 8: Визуализация =====

% 8.1 Сравнение предсказаний
figure('Name', 'Сравнение предсказаний', 'Position', [100, 100, 1200, 500]);

subplot(1,2,1);
scatter(actual, pred_fix, 20, 'b', 'filled', 'DisplayName', 'Baseline (фикс. парам.)');
hold on;
scatter(actual, pred_tune, 20, 'r', 'filled', 'DisplayName', 'Tunable (настр. парам.)');
plot([min(actual), max(actual)], [min(actual), max(actual)], 'k--', 'LineWidth', 1.5);
xlabel('Истинное значение'); ylabel('Предсказание');
title(sprintf('Scatter: Fix R2=%.3f vs Tune R2=%.3f', r2_fix, r2_tune));
legend('Location', 'best'); grid on; axis equal;

subplot(1,2,2);
plot(1:length(actual), residuals_fix, 'b-', 'LineWidth', 1, 'DisplayName', 'Остатки Baseline');
hold on;
plot(1:length(actual), residuals_tune, 'r-', 'LineWidth', 1, 'DisplayName', 'Остатки Tunable');
yline(0, 'k--');
xlabel('Номер примера'); ylabel('Остаток');
title('Сравнение остатков');
legend('Location', 'best'); grid on;

% 8.2 Гистограмма абсолютных ошибок
figure('Name', 'Распределение ошибок', 'Position', [100, 100, 900, 400]);
histogram(abs_err_fix, 20, 'FaceAlpha', 0.5, 'DisplayName', sprintf('Baseline (MAE=%.2f)', mae_fix));
hold on;
histogram(abs_err_tune, 20, 'FaceAlpha', 0.5, 'DisplayName', sprintf('Tunable (MAE=%.2f)', mae_tune));
xlabel('Абсолютная ошибка'); ylabel('Частота');
title('Распределение абсолютных ошибок');
legend('Location', 'best'); grid on;

%% ===== Часть 9: Градиентный бустинг (внешний baseline) =====

gbModel = fitrensemble(XTrainGB, yTrainGB, ...
    'Method', 'LSBoost', 'NumLearningCycles', 100, ...
    'Learners', 'tree', 'LearnRate', 0.1);
yPredGB = predict(gbModel, XTestGB);
mae_gb  = mean(abs(yPredGB - yTestGB));
r2_gb   = 1 - sum((yTestGB - yPredGB).^2) / sum((yTestGB - mean(yTestGB)).^2);

fprintf('\n========== ГРАДИЕНТНЫЙ БУСТИНГ (внешний baseline) ==========\n');
fprintf('GB  — MAE: %.4f, R2: %.4f\n', mae_gb, r2_gb);

%% ===== Часть 10: Итоговая сводка =====

fprintf('\n========== ИТОГОВАЯ СВОДКА ==========\n');
fprintf('Модель                   | MAE     | R2      | Параметры h/alpha\n');
fprintf('------------------------ | ------- | ------- | ----------------------------\n');
fprintf('Heating2DModel (fix)     | %.4f  | %.4f  | фиксированы (неверные)\n', mae_fix, r2_fix);
fprintf('Heating2DTunableModel    | %.4f  | %.4f  | настраиваются → к истинным\n', mae_tune, r2_tune);
fprintf('GradientBoosting         | %.4f  | %.4f  | нет физических параметров\n', mae_gb, r2_gb);
fprintf('\nВывод: настройка физических параметров даёт ');
if mae_tune < mae_fix
    fprintf('положительный эффект (MAE улучшился на %.1f%%).\n', (1 - mae_tune/mae_fix)*100);
else
    fprintf('не дала улучшения по MAE. Проверьте скорость обучения.\n');
end
if p_ttest < 0.05
    fprintf('Улучшение статистически значимо (p = %.4f).\n', p_ttest);
else
    fprintf('Улучшение НЕ является статистически значимым (p = %.4f).\n', p_ttest);
end

% Сохранение результатов
save('ExpTunable_Results.mat', 'results', 'actual', 'h_true', 'alpha_true', 'h_init', 'alpha_init');
fprintf('\nРезультаты сохранены в ExpTunable_Results.mat\n');