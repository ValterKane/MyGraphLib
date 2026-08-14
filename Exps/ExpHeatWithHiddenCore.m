%% Очистить все
clear; clc;
rng(22);

import BWGraph.*;
import BWGraph.CustomMatrix.*;
import BWGraph.RandomGenerator.*;
import BWGraph.Trainer.*;

HeatBC = coreFunctions.Heating2DModel(30, 20, 20, 70, 1.5e-5, 0.3, 0.360, 30, 10);

betaGen = FullRandomBetaGen(0,1); % Гиперпараметр

nodeA = Node(1, 1, 'Black', HeatBC, 'linear');  % v₁ — начало цепочки
nodeB = Node(2, 1, 'Black', HeatBC, 'linear');  % v₂ — середина
nodeC = Node(3, 1, 'White', HeatBC, 'linear');  % v₃ — выход

nodeA.addEdge(nodeB);  % A→B: ctx_B = F_A на stage 2
nodeB.addEdge(nodeC);  % B→C: ctx_C = F_B на stage 2


% Индивидуальные параметры для вершин (общие для всех экспериментов)
NodeWeight = [1 1 1]; % Весовые коэффициенты вершин

% Создаем графовую модель
modelShell = GraphShell(betaGen,NodeWeight,nodeA,nodeB,nodeC);

% Генерация данных и подготовка подвыборок (синтетика)
% Генерация данных с учетом индивидуальных характеристик вершин
numSamples = 100;
numOfNodes = numel(modelShell.ListOfNodes);
numOfWhiteNodes = modelShell.GetNumOfWhiteNode; % Получаем количество вершин
numInputParams = HeatBC.GetNumOfInputParams(); % Получаем количество входных параметров

% Генерация данных
timeRange_for_v1 = [1000, 11000];
TinfRange_for_v1 = [900, 1200];

timeRange_for_v2 = [1000, 15000];
TinfRange_for_v2 = [800, 1150];  

timeRange_for_v3 = [1000, 9000];
TinfRange_for_v3 = [900, 1100];   

% Случайные значения времени и температуры окружающей среды (значения X)
timeValues_for_v1 = rand(numSamples, 1) * (timeRange_for_v1(2) - timeRange_for_v1(1)) + timeRange_for_v1(1);
TinfValues_for_v1 = rand(numSamples, 1) * (TinfRange_for_v1(2) - TinfRange_for_v1(1)) + TinfRange_for_v1(1);

timeValues_for_v2 = rand(numSamples, 1) * (timeRange_for_v2(2) - timeRange_for_v2(1)) + timeRange_for_v2(1);
TinfValues_for_v2 = rand(numSamples, 1) * (TinfRange_for_v2(2) - TinfRange_for_v2(1)) + TinfRange_for_v2(1);

timeValues_for_v3 = rand(numSamples, 1) * (timeRange_for_v3(2) - timeRange_for_v3(1)) + timeRange_for_v3(1);
TinfValues_for_v3 = rand(numSamples, 1) * (TinfRange_for_v3(2) - TinfRange_for_v3(1)) + TinfRange_for_v3(1);


% Целевая переменная: взвешенная сумма выходов «истинных» солверов.
% Структурно представима графом — линейные члены (White/Black linear)
TavgValues = zeros(numSamples, 1);
for i = 1:numSamples
    inputParams_for_v1 = [timeValues_for_v1(i); TinfValues_for_v1(i)];
    inputParams_for_v2 = [timeValues_for_v2(i); TinfValues_for_v2(i)];
    inputParams_for_v3 = [timeValues_for_v3(i); TinfValues_for_v3(i)];

    Hidden_Core_1 = coreFunctions.Heating2DModel(30, 20, 20, 70, 1.5e-5, 0.3, 0.360, 30, 10);
    temp1 = Hidden_Core_1.CalcCoreFunction(inputParams_for_v1);
    Hidden_Core_2 = coreFunctions.Heating2DModel(30, 20, 20, 70, 1.5e-5, 0.3, 0.360, temp1, 10);
    temp2 = Hidden_Core_2.CalcCoreFunction(inputParams_for_v2);
    Hidden_Core_3 = coreFunctions.Heating2DModel(30, 20, 20, 70, 1.5e-5, 0.3, 0.360, temp2, 10);

    TavgValues(i) = Hidden_Core_3.CalcCoreFunction(inputParams_for_v3);
end

XData_for_bagging = [timeValues_for_v1, timeValues_for_v2, timeValues_for_v3, ...
        TinfValues_for_v1,TinfValues_for_v2,TinfValues_for_v3];
YData_for_bagging = TavgValues;

% Подготовим материалы
% Определим матрицы входа и выхода
XData = repmat(BWMatrix(), numSamples, 1);
YData = repmat(BWMatrix(), numSamples, 1);

for i = 1:numSamples
    inputParams_for_v1 = [timeValues_for_v1(i); TinfValues_for_v1(i)];
    inputParams_for_v2 = [timeValues_for_v2(i); TinfValues_for_v2(i)];
    inputParams_for_v3 = [timeValues_for_v3(i); TinfValues_for_v3(i)];
    XData(i) = XData(i).addRow(inputParams_for_v1);
    XData(i) = XData(i).addRow(inputParams_for_v2);
    XData(i) = XData(i).addRow(inputParams_for_v3);
end

for i = 1:numSamples
    yMatrix = zeros(1,numOfWhiteNodes);
    for j = 1:numOfWhiteNodes
        yMatrix(j) = TavgValues(i);
    end
    YData(i) = YData(i).addRow(yMatrix);
end

cv = cvpartition(size(XData,1), 'HoldOut', 0.3);
XDataTrain = XData(training(cv), :);
YDataTrain = YData(training(cv));
XDataTest = XData(test(cv), :);
YDataTest = YData(test(cv));

XTrain = XData_for_bagging(training(cv), :);
yTrain = YData_for_bagging(training(cv));
XTest = XData_for_bagging(test(cv), :);
yTest = YData_for_bagging(test(cv));

%% Phase 1: K=1 — обучение без контекста
fprintf('\n=== ФАЗА 1: K=1 ===\n');
trainerOptions1 = TrainingOptions( ...
    "LearningRate", 0.01, ...
    "NodeSize", [1, 1, 1], ...
    "LRDecayInterval", 50, ...
    "TargetError", 0.5, ...
    "Epoches", 15, ...
    "ClipUp", 1e7, ...
    "ClipDown", -1e7, ...
    "Lambda_Self", 0, ...
    "Lambda_Struct", 1, ...
    "ContextStages", 1, ...
    "EnablePlateauEscape", false, ...
    "ErrorMetric",'mae', ...
    "LossFunction",'mse', ...
    "TargetNodeIndices",[], ...
    "BatchSize", 1);

trainer = Trainer(modelShell, trainerOptions1);
trainer.Train(XDataTrain, YDataTrain, XDataTest, YDataTest);

%% Phase 2: K=3 — включаем контекст (NumStages и GammaCtx уже в модели)
fprintf('\n=== ФАЗА 2: K=3 ===\n');
trainerOptions2 = TrainingOptions( ...
    "LearningRate", 0.01, ...
    "NodeSize", [1, 1, 1], ...
    "LRDecayInterval", 50, ...
    "TargetError", 0.5, ...
    "Epoches", 25, ...
    "ClipUp", 1e7, ...
    "ClipDown", -1e7, ...
    "Lambda_Self", 0, ...
    "Lambda_Struct", 1, ...
    "ContextStages", 3, ...
    "EnablePlateauEscape", false, ...
    "ErrorMetric",'mae', ...
    "LossFunction",'mse', ...
    "TargetNodeIndices",[], ...
    "BatchSize", 1);

trainer2 = Trainer(modelShell, trainerOptions2);
trainer2.Train(XDataTrain, YDataTrain, XDataTest, YDataTest);

%% Тест на случайном лесу
numTrees = 6;

gbModel = fitrensemble(XTrain, yTrain, ...
    'Method', 'LSBoost', ...
    'NumLearningCycles', 100, ...
    'Learners', 'tree', ...
    'LearnRate', 0.1);

% Предсказание
yPredGB = predict(gbModel, XTest);

% Оценка качества
mseGB = mean((yPredGB - yTest).^2);
r2GB = 1 - sum((yTest - yPredGB).^2) / sum((yTest - mean(yTest)).^2);
fprintf('Градиентный бустинг - MSE: %.4f, R²: %.4f\n', mseGB, r2GB);

scatter(yTest, yPredGB);
hold on;
plot([min(yTest) max(yTest)], [min(yTest) max(yTest)], 'r--');
xlabel('Истинные значения');
ylabel('Предсказания GB');
title('Градиентный бустинг');
grid on;
%% Тестирование
numTestSamples = numel(YDataTest);
actualValue = zeros(1,numTestSamples);
predictionValue = zeros(1, numTestSamples);
index = 1;

for j = 1:numTestSamples
    modelShell.Forward(XDataTest(j));
    actualValue(j) = YDataTest(j).getRow(1);
    predicton = modelShell.GetModelResults();
    predictionValue(j) = predicton(3);
end

% Строим график сравнения
figure(...
    'Name', 'Тестирование', ...
    'Position', [10, 10, 900, 500], ...
    'Color', [0.95, 0.95, 0.95], ...
    'Resize', 'off' ...
);

hold on;

% Рисуем линии фактических и модельных значений
plot(1:numTestSamples, actualValue, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Фактические значения');
plot(1:numTestSamples, predictionValue, 'r--s', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Модельные значения');

% Настраиваем график
xlabel('Номер тестового примера');
ylabel('Значение');
title('Апробация модели на тестовом подмножестве');
legend('show', 'Location', 'best');

set(gca, 'FontSize', 14, 'FontWeight', 'bold');

grid on;
hold off;

residual = actualValue - predictionValue;

% Тест на нормальность (Lilliefors)
[h, p] = lillietest(residual);
if h == 0
    fprintf('Остатки нормально распределены (p=%.4f)\n', p);
else
    fprintf('Остатки НЕ нормальны (p=%.4f)\n', p);
end


residual = actualValue - predictionValue;

% 1. Сортируем данные по предсказанным значениям (или по одной из переменных)
[sorted_y_pred, sort_idx] = sort(predictionValue);
sorted_residuals = residual(sort_idx);

% 2. Разделяем остатки на 3 группы (исключая среднюю часть)
n = length(residual);
k = floor(n / 3); % Размер групп

% Первая группа (наименьшие ŷ)
residuals_low = sorted_residuals(1:k);

% Последняя группа (наибольшие ŷ)
residuals_high = sorted_residuals(end-k+1:end);

% 3. Сравниваем дисперсии (F-тест)
var_low = var(residuals_low);
var_high = var(residuals_high);

% F-статистика
F_stat = var_high / var_low; % Берем большую дисперсию в числитель

% Критическое значение F-распределения (для alpha=0.05)
df = k - 1;
F_critical = finv(0.95, df, df);

fprintf('F_stat = %.4f\n', F_stat);
fprintf('F_crit = %.4f\n', F_critical);
fprintf('P_value = %.4f\n', F_stat / F_critical);

% Проверка гипотезы
if F_stat > F_critical
    disp('Гетероскедастичность (p < 0.05)');
else
    disp('Гомоскедастичность (p > 0.05)');
end


SS_residual = sum((actualValue - predictionValue).^2);       
SS_total = sum((actualValue - mean(actualValue)).^2); 
R2 = 1 - (SS_residual / SS_total);  

fprintf('R² BW_Модель = %.4f\n', R2);

%% Валидация
validSamples = 50;
rng(1111);
% Генерация данных
timeRange_for_v1 = [1000, 11000];
TinfRange_for_v1 = [900, 1200];

timeRange_for_v2 = [1000, 15000];
TinfRange_for_v2 = [800, 1150];  

timeRange_for_v3 = [1000, 9000];
TinfRange_for_v3 = [900, 1100];  

% Случайные значения времени и температуры окружающей среды (значения X)
timeValues_for_v1 = rand(validSamples, 1) * (timeRange_for_v1(2) - timeRange_for_v1(1)) + timeRange_for_v1(1);
TinfValues_for_v1 = rand(validSamples, 1) * (TinfRange_for_v1(2) - TinfRange_for_v1(1)) + TinfRange_for_v1(1);

timeValues_for_v2 = rand(validSamples, 1) * (timeRange_for_v2(2) - timeRange_for_v2(1)) + timeRange_for_v2(1);
TinfValues_for_v2 = rand(validSamples, 1) * (TinfRange_for_v2(2) - TinfRange_for_v2(1)) + TinfRange_for_v2(1);

timeValues_for_v3 = rand(validSamples, 1) * (timeRange_for_v3(2) - timeRange_for_v3(1)) + timeRange_for_v3(1);
TinfValues_for_v3 = rand(validSamples, 1) * (TinfRange_for_v3(2) - TinfRange_for_v3(1)) + TinfRange_for_v3(1);


% Зададим синтетические значения средней температуры, смещенной
% относительно нагрева соседних вершин
TavgValues = zeros(numSamples, 1);
for i = 1:validSamples
     inputParams_for_v1 = [timeValues_for_v1(i); TinfValues_for_v1(i)];
    inputParams_for_v2 = [timeValues_for_v2(i); TinfValues_for_v2(i)];
    inputParams_for_v3 = [timeValues_for_v3(i); TinfValues_for_v3(i)];

    Hidden_Core_1 = coreFunctions.Heating2DModel(30, 20, 20, 70, 1.5e-5, 0.3, 0.360, 30, 10);
    temp1 = Hidden_Core_1.CalcCoreFunction(inputParams_for_v1);
    Hidden_Core_2 = coreFunctions.Heating2DModel(30, 20, 20, 70, 1.5e-5, 0.3, 0.360, temp1, 10);
    temp2 = Hidden_Core_2.CalcCoreFunction(inputParams_for_v2);
    Hidden_Core_3 = coreFunctions.Heating2DModel(30, 20, 20, 70, 1.5e-5, 0.3, 0.360, temp2, 10);

    TavgValues(i) = Hidden_Core_3.CalcCoreFunction(inputParams_for_v3);
end

XValidData = [timeValues_for_v1, timeValues_for_v2, timeValues_for_v3, ...
        TinfValues_for_v1,TinfValues_for_v2,TinfValues_for_v3];

% Определим матрицы входа и выхода
XDataValid = repmat(BWMatrix(), validSamples, 1);
YDataValid = repmat(BWMatrix(), validSamples, 1);

for i = 1:validSamples
    inputParams_for_v1 = [timeValues_for_v1(i); TinfValues_for_v1(i)];
    inputParams_for_v2 = [timeValues_for_v2(i); TinfValues_for_v2(i)];
    inputParams_for_v3 = [timeValues_for_v3(i); TinfValues_for_v3(i)];
    XDataValid(i) = XDataValid(i).addRow(inputParams_for_v1);
    XDataValid(i) = XDataValid(i).addRow(inputParams_for_v2);
    XDataValid(i) = XDataValid(i).addRow(inputParams_for_v3);
end

for i = 1:validSamples
    yMatrix = zeros(1,numOfWhiteNodes);
    for j = 1:numOfWhiteNodes
        yMatrix(j) = TavgValues(i);
    end
    YDataValid(i) = YDataValid(i).addRow(yMatrix);
end

validValue = zeros(1,validSamples);
predictOnValid = zeros(1, validSamples);
predictOnGBModel = zeros(1,validSamples);

for j = 1:validSamples
    forecast = modelShell.GetCurrentResult(XDataValid(j));
    predictOnValid(j) = forecast(3);
    validValue(j) = TavgValues(j);
    predictOnGBModel(j) = predict(gbModel,XValidData(j,:));
end

% Строим график сравнения
figure(...
    'Name', 'Валидация', ...
    'Position', [10, 10, 900, 500], ...
    'Color', [0.95, 0.95, 0.95], ...
    'Resize', 'off' ...
);
hold on;

% Рисуем линии фактических и модельных значений
plot(1:validSamples, validValue, 'k-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Фактические значения');
plot(1:validSamples, predictOnValid, 'r-*', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Предложенная мета-архитектура');
plot(1:validSamples, predictOnGBModel, 'b-*', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'Градиентный бустинг');

% Настраиваем график
xlabel('Номер примера');
ylabel('Значение');
title('Сравнение предложенного оператора композиции и LSBoost');
legend('show', 'Location', 'best');

set(gca, 'FontSize', 14, 'FontWeight', 'bold');

grid on;
hold off;

residual = validValue - predictOnValid;

% Тест на нормальность (Lilliefors)
[h, p] = lillietest(residual);
if h == 0
    fprintf('Остатки нормально распределены (p=%.4f)\n', p);
else
    fprintf('Остатки НЕ нормальны (p=%.4f)\n', p);
end


residual = validValue - predictOnValid;

% 1. Сортируем данные по предсказанным значениям (или по одной из переменных)
[sorted_y_pred, sort_idx] = sort(predictOnValid);
sorted_residuals = residual(sort_idx);

% 2. Разделяем остатки на 3 группы (исключая среднюю часть)
n = length(residual);
k = floor(n / 3); % Размер групп

% Первая группа (наименьшие ŷ)
residuals_low = sorted_residuals(1:k);

% Последняя группа (наибольшие ŷ)
residuals_high = sorted_residuals(end-k+1:end);

% 3. Сравниваем дисперсии (F-тест)
var_low = var(residuals_low);
var_high = var(residuals_high);

% F-статистика
F_stat = var_high / var_low; % Берем большую дисперсию в числитель

% Критическое значение F-распределения (для alpha=0.05)
df = k - 1;
F_critical = finv(0.95, df, df);

fprintf('F_stat = %.4f\n', F_stat);
fprintf('F_crit = %.4f\n', F_critical);
fprintf('P_value = %.4f\n', F_stat / F_critical);

% Проверка гипотезы
if F_stat > F_critical
    disp('Гетероскедастичность (p < 0.05)');
else
    disp('Гомоскедастичность (p > 0.05)');
end


SS_residual = sum((validValue - predictOnValid).^2);       
SS_total = sum((validValue - mean(validValue)).^2); 
R2 = 1 - (SS_residual / SS_total);  

fprintf('R² BW_Модель = %.4f\n', R2);
    
SS_residual = sum((validValue - predictOnGBModel).^2);       
SS_total = sum((validValue - mean(validValue)).^2); 
R2 = 1 - (SS_residual / SS_total);

fprintf('R² GB_Модель = %.4f\n', R2);

fprintf('Результирующее MAE для BW %2.3f\n', mae(predictOnValid,validValue))
fprintf('Результирующее MAE для GB %2.3f\n', mae(predictOnGBModel,validValue))
%% Чек модели
diagnostics_BW = plotHeatingPrediction(validValue,predictOnValid);
%% Чек LSBoost (xGBoost на деревьях с МНК)
diagnostics_LSBoost = plotHeatingPrediction(validValue, predictOnGBModel);
%% Анализ метрики MAE
fprintf('Результирующее MAE на валидации %2.3f\n', mae(predictOnValid,validValue))
fprintf('Результирующее MAE на тесте %2.3f\n', mae(predictionValue,actualValue))

%% Анализ дисперсий
fprintf('D[Y_val]: %3f\n', var(predictOnValid))
fprintf('D[Y_val_actual] %3f\n', var(validValue))
fprintf('D[Y_test]: %3f\n', var(predictionValue))
fprintf('D[Y_test_actual]: %3f\n', var(actualValue))

%% Тест на равенство дисперсий на валидации
% Проводим F-тест
[h, p, ci, stats] = vartest2(predictOnValid, validValue);

fprintf('h = %d (1 - дисперсии не равны, 0 - равны)\n', h);
fprintf('p-value = %.4f\n', p);
fprintf('Отношение дисперсий (x/y) = %.4f\n', stats.fstat);
fprintf('Доверительный интервал: [%.4f, %.4f]\n', ci(1), ci(2));
%% Тест на равенство дисперсий на тестовой
% Проводим F-тест
[h, p, ci, stats] = vartest2(predictionValue, actualValue);

fprintf('h = %d (1 - дисперсии не равны, 0 - равны)\n', h);
fprintf('p-value = %.4f\n', p);
fprintf('Отношение дисперсий (x/y) = %.4f\n', stats.fstat);
fprintf('Доверительный интервал: [%.4f, %.4f]\n', ci(1), ci(2));

%%
function diagnostics = plotHeatingPrediction(y_true, y_pred, time_vector, model_name)
% Функция визуальной диагностики модели прогнозирования нагрева
% 
% Входные параметры:
%   y_true     - вектор реальных значений температуры
%   y_pred     - вектор предсказанных значений температуры
%   time_vector - вектор времени/индексов (опционально)
%   model_name - название модели для заголовков (опционально)
%
% Выходные параметры:
%   diagnostics - структура с метриками и результатами проверок

    % Проверка входных аргументов
    if nargin < 3
        time_vector = 1:length(y_true);
    end
    if nargin < 4
        model_name = 'Модель';
    end
    
    % Преобразуем в векторы-столбцы для надежности
    y_true = y_true(:);
    y_pred = y_pred(:);
    time_vector = time_vector(:);
    
    % Убедимся, что длины совпадают
    assert(length(y_true) == length(y_pred), 'Длины y_true и y_pred должны совпадать');
    assert(length(y_true) == length(time_vector), 'Длины y_true и time_vector должны совпадать');
    
    % Создаем фигуру с 4 субплотами
    figure('Position', [100, 100, 1400, 900]);
    
    % 1. Scatter plot: Предсказания vs Реальность
    subplot(2, 3, 1);
    plot(y_true, y_pred, 'b.', 'MarkerSize', 8);
    hold on;
    
    % Линия идеального предсказания (y = x)
    min_val = min([y_true; y_pred]);
    max_val = max([y_true; y_pred]);
    plot([min_val, max_val], [min_val, max_val], 'r-', 'LineWidth', 2);
    
    % Линия регрессии для выявления тренда
    coeffs = polyfit(y_true, y_pred, 1);
    y_fit = polyval(coeffs, [min_val, max_val]);
    plot([min_val, max_val], y_fit, 'g--', 'LineWidth', 1.5);
    
    xlabel('Реальная температура (°C)');
    ylabel('Предсказанная температура (°C)');
    title(sprintf('%s: Scatter Plot\nКрасный - идеал, Зеленый - факт', model_name));
    legend('Предсказания', 'Идеал', 'Тренд', 'Location', 'best');
    grid on;
    axis equal;
    
    % Добавляем текст с наклоном тренда
    text(min_val + 0.05*(max_val-min_val), max_val - 0.1*(max_val-min_val), ...
         sprintf('Наклон тренда: %.2f', coeffs(1)), 'FontSize', 10);
    
    % 2. Временной ряд: Реальность vs Предсказания
    subplot(2, 3, 2);
    plot(time_vector, y_true, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Реальная');
    hold on;
    plot(time_vector, y_pred, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Предсказанная');
    
    % Средняя температура
    y_mean = mean(y_true) * ones(size(time_vector));
    plot(time_vector, y_mean, 'g-', 'LineWidth', 1, 'DisplayName', 'Средняя');
    
    xlabel('Время/Индекс');
    ylabel('Температура (°C)');
    title('Временной ряд: Реальность vs Предсказания');
    legend('Location', 'best');
    grid on;
    
    % 3. График ошибок
    subplot(2, 3, 3);
    errors = y_pred - y_true;
    plot(time_vector, errors, 'b-', 'LineWidth', 1);
    hold on;
    plot(time_vector, zeros(size(time_vector)), 'r--', 'LineWidth', 1);
    
    % Добавляем доверительные интервалы
    std_err = std(errors);
    plot(time_vector, 2*std_err * ones(size(time_vector)), 'g--', 'LineWidth', 0.5);
    plot(time_vector, -2*std_err * ones(size(time_vector)), 'g--', 'LineWidth', 0.5);
    
    xlabel('Время/Индекс');
    ylabel('Ошибка (°C)');
    title(sprintf('Ошибка предсказания (MAE = %.2f°C)', mean(abs(errors))));
    legend('Ошибка', 'Ноль', '±2σ', 'Location', 'best');
    grid on;
    
    % 4. Гистограмма ошибок
    subplot(2, 3, 4);
    histogram(errors, 30, 'Normalization', 'pdf', 'FaceColor', [0.8 0.8 1]);
    hold on;
    
    % Нормальное распределение для сравнения
    x_range = linspace(min(errors), max(errors), 100);
    y_norm = normpdf(x_range, mean(errors), std(errors));
    plot(x_range, y_norm, 'r-', 'LineWidth', 2);
    
    xlabel('Ошибка (°C)');
    ylabel('Плотность');
    title('Распределение ошибок');
    legend('Ошибки', 'Нормальное', 'Location', 'best');
    grid on;
    
    % 5. Автокорреляция ошибок (важно для временных рядов)
    subplot(2, 3, 5);
    [acf, lags] = xcorr(errors - mean(errors), 20, 'normalized');
    lags = lags(21:end);  % Берем только положительные лаги
    acf = acf(21:end);
    
    stem(lags, acf, 'filled', 'LineWidth', 1.5);
    hold on;
    
    % Доверительные интервалы
    conf_level = 1.96 / sqrt(length(errors));
    plot([0, 20], [conf_level, conf_level], 'r--');
    plot([0, 20], [-conf_level, -conf_level], 'r--');
    
    xlabel('Лаг');
    ylabel('Автокорреляция');
    title('Автокорреляция ошибок');
    grid on;
    ylim([-1, 1]);
    
    % 6. Остатки vs Предсказания (гомоскедастичность)
    subplot(2, 3, 6);
    plot(y_pred, errors, 'b.', 'MarkerSize', 8);
    hold on;
    plot([min(y_pred), max(y_pred)], [0, 0], 'r--', 'LineWidth', 1);
    
    % Скользящее среднее для выявления гетероскедастичности
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
    
    % Общий заголовок
    sgtitle(sprintf('Диагностика модели: %s', model_name), 'FontSize', 14, 'FontWeight', 'bold');
    
    % Расчет метрик
    diagnostics = calculateMetrics(y_true, y_pred);
    
    % Вывод метрик в командное окно
    fprintf('\n========== МЕТРИКИ МОДЕЛИ: %s ==========\n', model_name);
    fprintf('R² (коэффициент детерминации): %.4f\n', diagnostics.R2);
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
    
    % Диагностические выводы
    printDiagnostics(diagnostics);
end

function metrics = calculateMetrics(y_true, y_pred)
    % Расчет основных метрик
    n = length(y_true);
    errors = y_pred - y_true;
    abs_errors = abs(errors);
    
    % R²
    ss_res = sum(errors.^2);
    ss_tot = sum((y_true - mean(y_true)).^2);
    metrics.R2 = 1 - ss_res/ss_tot;
    
    % MAE, RMSE, MAPE, MedAE
    metrics.MAE = mean(abs_errors);
    metrics.RMSE = sqrt(mean(errors.^2));
    
    % MAPE с защитой от деления на ноль
    non_zero_idx = y_true ~= 0;
    if any(non_zero_idx)
        metrics.MAPE = mean(abs_errors(non_zero_idx) ./ abs(y_true(non_zero_idx))) * 100;
    else
        metrics.MAPE = NaN;
    end
    
    metrics.MedAE = median(abs_errors);
    
    % Статистика целевой переменной
    metrics.y_mean = mean(y_true);
    metrics.y_std = std(y_true);
    metrics.y_min = min(y_true);
    metrics.y_max = max(y_true);
    metrics.y_range = metrics.y_max - metrics.y_min;
    
    % Дополнительные метрики
    metrics.max_error = max(abs_errors);
    metrics.error_std = std(errors);
end

function printDiagnostics(diagnostics)
    % Вывод диагностических сообщений
    fprintf('========== ДИАГНОСТИЧЕСКИЕ ВЫВОДЫ ==========\n');
    
    % Проверка R²
    if diagnostics.R2 < 0
        fprintf('ПРОБЛЕМА: R² = %.2f (отрицательный!)\n', diagnostics.R2);
        fprintf('   Модель работает ХУЖЕ, чем просто предсказание среднего.\n');
        if diagnostics.R2 < -0.1
            fprintf('   Возможно, модель предсказывает в противофазе с реальностью.\n');
        end
    elseif diagnostics.R2 < 0.3
        fprintf('R² = %.2f (низкий)\n', diagnostics.R2);
        fprintf('   Модель объясняет только %.0f%% вариации данных.\n', diagnostics.R2*100);
    elseif diagnostics.R2 < 0.7
        fprintf('R² = %.2f (средний)\n', diagnostics.R2);
        fprintf('   Модель объясняет %.0f%% вариации данных.\n', diagnostics.R2*100);
    else
        fprintf('R² = %.2f (отличный!)\n', diagnostics.R2);
    end
    
    % Проверка соотношения RMSE и стандартного отклонения
    rmse_std_ratio = diagnostics.RMSE / diagnostics.y_std;
    if rmse_std_ratio > 1
        fprintf('RMSE (%.2f) БОЛЬШЕ стандартного отклонения (%.2f)\n', ...
                diagnostics.RMSE, diagnostics.y_std);
        fprintf('   Это объясняет низкий R².\n');
    elseif rmse_std_ratio > 0.7
        fprintf('RMSE составляет %.0f%% от стандартного отклонения\n', rmse_std_ratio*100);
    else
        fprintf('RMSE составляет %.0f%% от стандартного отклонения\n', rmse_std_ratio*100);
    end
    
    % Проверка размаха данных
    if diagnostics.MAE > 0.2 * diagnostics.y_range
        fprintf('MAE (%.2f) составляет >20%% от размаха данных (%.2f)\n', ...
                diagnostics.MAE, diagnostics.y_range);
        fprintf('   Это большая относительная ошибка.\n');
    end
    
    % Рекомендации
    fprintf('\n🔍 РЕКОМЕНДАЦИИ:\n');
    if diagnostics.R2 < 0
        fprintf('   - Проверьте, нет ли перепутанных меток (переменных)\n');
        fprintf('   - Проверьте выбросы в данных\n');
        fprintf('   - Попробуйте инвертировать предсказания для теста\n');
    elseif diagnostics.R2 < 0.3
        fprintf('   - Добавьте больше признаков (историю нагрева)\n');
        fprintf('   - Проверьте лаги (возможно, температура зависит от предыдущих значений)\n');
        fprintf('   - Попробуйте другую модель (Random Forest, XGBoost)\n');
    end
    
     fprintf('==============================================\n');
end