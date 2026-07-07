%% Очистить все
clear; clc;
rng(22);

import BWGraph.*;
import BWGraph.CustomMatrix.*;
import BWGraph.RandomGenerator.*;
import BWGraph.Trainer.*;

HeatBC = coreFunctions.Heating2DModel(30, 20, 20, 70, 1.5e-5, 0.3, 0.360, 30, 10);
betaGen = FullRandomBetaGen(0,1); % Гиперпараметр

nodeA = Node(1, 1,'White',HeatBC,'linear');
nodeB = Node(2, 1,'Black',HeatBC,'linear');
nodeC = Node(3, 1,'Black',HeatBC,'linear');

nodeB.addEdge(nodeA);
nodeC.addEdge(nodeA);

% nodeA.addEdge(nodeB);
% nodeB.addEdge(nodeA);
% 
% nodeA.addEdge(nodeC);
% nodeC.addEdge(nodeA);
% 
% nodeB.addEdge(nodeC);
% % nodeC.addEdge(nodeB);


% Индивидуальные параметры для вершин (общие для всех экспериментов)
NodeWeight = [1 0.5 0.5]; % Весовые коэффициенты вершин

% Создаем графовую модель
modelShell = GraphShell(betaGen,NodeWeight,nodeA,nodeB,nodeC);

% Генерация данных и подготовка подвыборок (синтетика)
% Генерация данных с учетом индивидуальных характеристик вершин
numSamples = 50;
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


% Зададим синтетические значения средней температуры, смещенной
% относительно нагрева соседних вершин
TavgValues = zeros(numSamples, 1);
for i = 1:numSamples
    inputParams_for_v1 = [timeValues_for_v1(i); TinfValues_for_v1(i)];
    inputParams_for_v2 = [timeValues_for_v2(i); TinfValues_for_v2(i)];
    inputParams_for_v3 = [timeValues_for_v3(i); TinfValues_for_v3(i)];

    TavgValues(i) = HeatBC.CalcCoreFunction(inputParams_for_v1) + ...
        0.5 * HeatBC.CalcCoreFunction(inputParams_for_v2) + ...
        0.3 * HeatBC.CalcCoreFunction(inputParams_for_v3);
    
end

XData_for_bagging = [timeValues_for_v1, timeValues_for_v2, timeValues_for_v3, ...
        TinfValues_for_v1,TinfValues_for_v2,TinfValues_for_v3];
YData_for_bagging = TavgValues;

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

%% Настройка учителя
% Опции настройки
trainerOptions = TrainingOptions( ...
    "LearningRate", 0.01, ...
    "NodeSize", [1, 1, 1], ...
    "LRDecayInterval", 50, ...
    "TargetError", 0.5, ...
    "Epoches", 500, ...
    "ClipUp", 1e7, ...
    "ClipDown", -1e7, ...
    "Lambda_Self", 0, ...
    "Lambda_Struct", 1, ...
    "EnablePlateauEscape", false, ...
    "ErrorMetric",'mae', ...
    "LossFunction",'mse', ...
    "TargetNodeIndices",[], ...
    "BatchSize", 1);

% Инициализация учителя
trainer = Trainer(modelShell, trainerOptions);

%% Запуск процесса
trainer.Train(XDataTrain, YDataTrain, XDataTest, YDataTest);

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
    predictionValue(j) = predicton(1);
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
    
    TavgValues(i) = HeatBC.CalcCoreFunction(inputParams_for_v1) + ...
        0.1 * HeatBC.CalcCoreFunction(inputParams_for_v2) + ...
        0.15 *HeatBC.CalcCoreFunction(inputParams_for_v3);
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
    predictOnValid(j) = forecast(1);
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
function diagnoseRFNaN(XTrain, yTrain, XTest)
    fprintf('=== ДИАГНОСТИКА ПРОБЛЕМЫ NaN В СЛУЧАЙНОМ ЛЕСЕ ===\n');
    
    % 1. Проверка данных
    fprintf('\n1. АНАЛИЗ ДАННЫХ:\n');
    fprintf('   Пропуски в X: %d\n', sum(isnan(XTrain(:))));
    fprintf('   Пропуски в y: %d\n', sum(isnan(yTrain)));
    fprintf('   Inf в X: %d\n', sum(isinf(XTrain(:))));
    fprintf('   Inf в y: %d\n', sum(isinf(yTrain)));
    
    % 2. Проверка уникальности
    fprintf('\n2. УНИКАЛЬНОСТЬ ДАННЫХ:\n');
    for i = 1:size(XTrain,2)
        n_unique = length(unique(XTrain(:,i)));
        fprintf('   Признак %d: %d уникальных значений\n', i, n_unique);
        if n_unique < 5
            fprintf('     ВНИМАНИЕ: Мало уникальных значений в признаке %d\n', i);
        end
    end
    
    % 3. Проверка выбросов
    fprintf('\n3. ВЫБРОСЫ:\n');
    z_scores = abs((yTrain - mean(yTrain)) / std(yTrain));
    outliers = z_scores > 3;
    fprintf('   Выбросы в y: %d (%.1f%%)\n', sum(outliers), 100*mean(outliers));
    
    % 4. Тестирование разных параметров
    fprintf('\n4. ТЕСТИРОВАНИЕ ПАРАМЕТРОВ:\n');
    
    param_combinations = [
        1, 1;    % MinLeafSize=1, NumTrees=10
        5, 50;   % MinLeafSize=5, NumTrees=50
        10, 100; % MinLeafSize=10, NumTrees=100
        20, 200; % MinLeafSize=20, NumTrees=200
    ];
    
    for i = 1:size(param_combinations, 1)
        min_leaf = param_combinations(i, 1);
        num_trees = param_combinations(i, 2);
        
        try
            rf_test = TreeBagger(num_trees, XTrain, yTrain, ...
                'Method', 'regression', ...
                'MinLeafSize', min_leaf, ...
                'NumPredictorsToSample', ceil(sqrt(size(XTrain,2))));
            
            pred_test = predict(rf_test, XTest);
            pred_test = str2double(pred_test);
            
            if any(isnan(pred_test))
                fprintf('   MinLeaf=%d, Trees=%d: ЕСТЬ NaN\n', min_leaf, num_trees);
            else
                fprintf('   MinLeaf=%d, Trees=%d: OK\n', min_leaf, num_trees);
            end
        catch ME
            fprintf('   MinLeaf=%d, Trees=%d: ОШИБКА - %s\n', min_leaf, num_trees, ME.message);
        end
    end
    
    % 5. Проверка на мультиколлинеарность
    fprintf('\n5. МУЛЬТИКОЛЛИНЕАРНОСТЬ:\n');
    if size(XTrain,2) > 1
        corr_matrix = corrcoef(XTrain);
        corr_matrix(logical(eye(size(corr_matrix)))) = 0;
        [max_corr, idx] = max(abs(corr_matrix(:)));
        if max_corr > 0.95
            fprintf('   Обнаружена высокая корреляция (%.3f) между признаками\n', max_corr);
        else
            fprintf('   Корреляция в норме (макс=%.3f)\n', max_corr);
        end
    end
    
    % 6. Рекомендации
    fprintf('\n6. РЕКОМЕНДАЦИИ:\n');
    fprintf('   - Увеличьте MinLeafSize (попробуйте 5-20)\n');
    fprintf('   - Уменьшите количество деревьев\n');
    fprintf('   - Проверьте на наличие константных признаков\n');
    fprintf('   - Удалите выбросы из обучающей выборки\n');
    fprintf('   - Нормализуйте признаки\n');
end