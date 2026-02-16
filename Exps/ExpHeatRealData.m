%% Отрисовать граф
modelShell.DrawGraph_New('Модель нагрева');

%% Загрузка реальных данных
data = readtable('C:\Users\darkd\Desktop\2024-2025\Математическая модель многозонной печи\Готовые данные по первой садке.xlsx');
numOfWhiteNodes = modelShell.GetNumOfWhiteNode; % Получаем количество вершин
numSamples = height(data);

% Определим матрицы входа и выхода
XData = repmat(BWMatrix(), numSamples, 1);
YData = repmat(BWMatrix(), numSamples, 1);

timeValues_for_v1 = (data{:,'H12'} * 60 + data{:,'M12'})*60;
TinfValues_for_v1 = (data{:,'minT12'} + data{:, 'maxT12'})/2;

timeValues_for_v2 = (data{:,'H34'} * 60 + data{:,'M34'})*60;
TinfValues_for_v2 = (data{:,'minT34'} + data{:, 'maxT34'})/2;

timeValues_for_v3 = (data{:,'H56'} * 60 + data{:,'M56'})*60;
TinfValues_for_v3 = (data{:,'minT56'} + data{:, 'maxT56'})/2;

for i = 1:numSamples
    inputParams_for_v1 = [timeValues_for_v1(i); TinfValues_for_v1(i)];
    inputParams_for_v2 = [timeValues_for_v2(i); TinfValues_for_v2(i)];
    inputParams_for_v3 = [timeValues_for_v3(i); TinfValues_for_v3(i)];
    XData(i) = XData(i).addRow(inputParams_for_v1);
    XData(i) = XData(i).addRow(inputParams_for_v2);
    XData(i) = XData(i).addRow(inputParams_for_v1);
    XData(i) = XData(i).addRow(inputParams_for_v2);
    XData(i) = XData(i).addRow(inputParams_for_v3);
end

resValue_for_v3 = (data{:,'maxTdou'} + data{:, 'minTduo'})/2;
% resValue_for_v3 = data{:,'maxTdou'};

for i = 1:numSamples
    yMatrix = zeros(1,numOfWhiteNodes);
    yMatrix(1) = resValue_for_v3(i);
    YData(i) = YData(i).addRow(yMatrix);
end

indices = randperm(numSamples);
splitPoint = round(0.7 * numSamples);
trainIndices = indices(1:splitPoint);
testIndices = indices(splitPoint+1:end);
 
% Определим обучающую и тестовую выборку
XDataTrain = XData(trainIndices);
YDataTrain = YData(trainIndices);
XDataTest = XData(testIndices);
YDataTest = YData(testIndices);

%% Загрузка данных валидации
if ~exist("validData", 'var')
    validData = readtable("DataForCompr.xlsx");
end

numOfNodes = numel(modelShell.ListOfNodes);
numOfWhiteNodes = modelShell.GetNumOfWhiteNode; % Получаем количество вершин
numInputParams = HeatBC_1.GetNumOfInputParams(); % Получаем количество входных параметров
numSamples = 131;

% Определим матрицы входа и выхода
XValidData = repmat(BWMatrix(), numSamples, 1);
YValidData = repmat(BWMatrix(), numSamples, 1);

t1 = table2array(validData(1:131,"F12_TimeDiff"));
t2 = table2array(validData(1:131,"F34_TimeDiff"));
t3 = table2array(validData(1:131,"F56_TimeDiff"));
tc = table2array(validData(1:131,"F_AF"));
T1 = (validData{1:131,'F12_TL'} + validData{1:131, 'F12_TR'})/2;
T2 = (validData{1:131,'F34_TL'} + validData{1:131, 'F34_TR'})/2;
T3 = (validData{1:131,'F56_TL'} + validData{1:131, 'F56_TR'})/2;
T_y = validData{1:131,'Tmax'}; 

data1 = [t1, T1];
data2 = [t2, T2];
data3 = [t3, T3];

for i = 1:numSamples
    XValidData(i) = XValidData(i).addRow(data1(i,:));
    XValidData(i) = XValidData(i).addRow(data2(i,:));
    XValidData(i) = XValidData(i).addRow(data1(i,:));
    XValidData(i) = XValidData(i).addRow(data2(i,:));
    XValidData(i) = XValidData(i).addRow(data3(i,:));
end

for i = 1:numSamples
    YValidData(i) = YValidData(i).addRow(T_y(i,:));
end

%% Проверка на валидации

numTestSamples = size(XValidData,1);

for i = 1:numTestSamples
    act(1,i) = YValidData(i).getRow(1);
    result = modelShell.GetCurrentResult(XValidData(i));
    predModel(1,i) = result(5);
end

mae_predModel = sum(abs(act - predModel)) / numel(act);

fprintf("МАЕ по стабилизированной модели = %.2f\n", mae_predModel);

% Строим график сравнения
figure(...
    'Name', 'Тестирование', ...
    'Position', [10, 10, 900, 500], ...
    'Color', [0.95, 0.95, 0.95], ...
    'Resize', 'off' ...
);

hold on;

% Рисуем линии фактических и модельных значений
plot(1:numTestSamples, act, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Фактические значения');
plot(1:numTestSamples, predModel, 'r--s', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Модельные значения');

% Настраиваем график
xlabel('Номер тестового примера');
ylabel('Значение');
title('Апробация модели на тестовом подмножестве');
legend('show', 'Location', 'best');

set(gca, 'FontSize', 14, 'FontWeight', 'bold');

grid on;
hold off;


%% Fast Start
clear; clc; close all;
rng(1111);
import BWGraph.*;
import BWGraph.CustomMatrix.*;
import BWGraph.RandomGenerator.*;
import BWGraph.Trainer.*;

HeatBC_1 = coreFunctions.Heating2DModel(30, 21, 21, 50, 1.5e-5, 0.3, 0.360, 30, 10);
HeatBC_2 = coreFunctions.Heating2DModel(60, 21, 21, 60, 1.5e-5, 0.3, 0.360, 500, 10);
HeatBC_3 = coreFunctions.Heating2DModel(90, 21, 21, 70, 1.5e-5, 0.3, 0.360, 1000, 10);

LinearTemper = coreFunctions.LinearFunction();

alfaGen = FullRandomAlfaGen(1,1e2);
betaGen = FullRandomBetaGen(1,1e3);

nodeA = Node(1, 30,'Black',HeatBC_1);
nodeB = Node(2, 30,'Black',HeatBC_2);
nodeC = Node(3, 30,'Black',HeatBC_1);
% nodeD = Node(4, 30,'Black',LinearTemper);
nodeE = Node(5, 30,'White',HeatBC_3);


nodeA.addEdge(nodeB);
nodeB.addEdge(nodeA);

nodeB.addEdge(nodeE);
nodeE.addEdge(nodeB);

nodeC.addEdge(nodeA);
nodeC.addEdge(nodeB);

% nodeD.addEdge(nodeB);
% nodeD.addEdge(nodeE);

% Создаем графовую модель
modelShell = GraphShell(alfaGen,betaGen,nodeA,nodeB,nodeC, nodeE);

% Индивидуальные параметры для вершин (общие для всех экспериментов)
NodeSize = [1.3 1.3 1.3, 1.3, 1.3]; % Коэффициенты 
NodeWeight = [1 1 1 1 1]; % Весовые коэффициенты вершин
%%
if ~exist("data", 'var')
    data = readtable("FirstPlaceData.xlsx");
end
numOfWhiteNodes = modelShell.GetNumOfWhiteNode; % Получаем количество вершин
numSamples = height(data);

% Определим матрицы входа и выхода
XData = repmat(BWMatrix(), numSamples, 1);
YData = repmat(BWMatrix(), numSamples, 1);

timeValues_for_v1 = (data{:,'H12'} * 60 + data{:,'M12'})*60;
TinfValues_for_v1 = (data{:,'minT12'} + data{:, 'maxT12'})/2;

timeValues_for_v2 = (data{:,'H34'} * 60 + data{:,'M34'})*60;
TinfValues_for_v2 = (data{:,'minT34'} + data{:, 'maxT34'})/2;

timeValues_for_v3 = (data{:,'H56'} * 60 + data{:,'M56'})*60;
TinfValues_for_v3 = (data{:,'minT56'} + data{:, 'maxT56'})/2;

for i = 1:numSamples
    inputParams_for_v1 = [timeValues_for_v1(i); TinfValues_for_v1(i)];
    inputParams_for_v2 = [timeValues_for_v2(i); TinfValues_for_v2(i)];
    inputParams_for_v3 = [timeValues_for_v3(i); TinfValues_for_v3(i)];
    XData(i) = XData(i).addRow(inputParams_for_v1);
    XData(i) = XData(i).addRow(inputParams_for_v2);
    % XData(i) = XData(i).addRow(inputParams_for_v1);
    XData(i) = XData(i).addRow(inputParams_for_v3);
end

resValue_for_v3 = (data{:,'maxTdou'} + data{:, 'minTduo'})/2;
% resValue_for_v3 = data{:,'maxTdou'};

for i = 1:numSamples
    yMatrix = zeros(1,numOfWhiteNodes);
    yMatrix(1) = resValue_for_v3(i);
    YData(i) = YData(i).addRow(yMatrix);
end

indices = randperm(numSamples);
splitPoint = round(0.7 * numSamples);
trainIndices = indices(1:splitPoint);
testIndices = indices(splitPoint+1:end);
 
% Определим обучающую и тестовую выборку
XDataTrain = XData(trainIndices);
YDataTrain = YData(trainIndices);
XDataTest = XData(testIndices);
YDataTest = YData(testIndices);
%% 
if ~exist("data", 'var')
    data = readtable("cleaned_data_for_compr.xlsx");
end

totalBatch = 300;

% Получение исходных данных
t1 = table2array(data(1:totalBatch,"F12_TimeDiff"));
t2 = table2array(data(1:totalBatch,"F34_TimeDiff"));
t3 = table2array(data(1:totalBatch,"F56_TimeDiff"));

T1 = (data{1:totalBatch,'F12_TL'} + data{1:totalBatch, 'F12_TR'})/2;
T2 = (data{1:totalBatch,'F34_TL'} + data{1:totalBatch, 'F34_TR'})/2;
T3 = (data{1:totalBatch,'F56_TL'} + data{1:totalBatch, 'F56_TR'})/2;
T_y = data{1:totalBatch,'Tmax'};

fprintf('Новый размер выборки: %d\n', totalBatch);

data_for_one = [t1, T1];
data_for_two = [t2, T2];
data_for_three = [t3, T3];

% Определим матрицы входа и выхода
XData = repmat(BWMatrix(), totalBatch, 1);
YData = repmat(BWMatrix(), totalBatch, 1);

for i = 1:totalBatch
    XData(i) = XData(i).addRow(data_for_one(i,:));
    XData(i) = XData(i).addRow(data_for_two(i,:));
    XData(i) = XData(i).addRow(data_for_one(i,:));
    XData(i) = XData(i).addRow(data_for_three(i,:));
end

for i = 1:totalBatch
    YData(i) = YData(i).addRow(T_y(i,:));
end

indices = randperm(totalBatch);
splitPoint = round(0.7 * totalBatch);
trainIndices = indices(1:splitPoint);
testIndices = indices(splitPoint+1:end);
 
% Определим обучающую и тестовую выборку
XDataTrain = XData(trainIndices);
YDataTrain = YData(trainIndices);
XDataTest = XData(testIndices);
YDataTest = YData(testIndices);

%% Настройка учителя
% Опции настройки
trainerOptions = TrainingOptions("LearningRate", 0.01, ...
    "Beta1", 0.9, ...
    "Beta2", 0.999, ...
    "Eps", 1e-8, ...
    "NodeSize", [1.3, 1.3, 1.3, 1.3, 1.3], ...
    "NodeWeight", [1,1,1,1,1], ...
    "Epoches", 500, ...
    "ClipUp", 1e15, ...
    "ClipDown", -1e15, ...
    "TargetError", 10, ...
    "Lambda_Agg", 1, ...
    "Lambda_Alph", 0.3, ...
    "Lambda_Beta", 0.3, ...
    "Lambda_Gamma",0.3, ...
    "ErrorMetric",'mae', ...
    "LossFunction",'mse', ...
    "TargetNodeIndices",[]);

% Инициализация учителя
trainer = Trainer(modelShell, trainerOptions);

%% Запуск процесса
trainer.Train(XDataTrain, YDataTrain, XDataTest, YDataTest);

%%
numTestSamples = size(XDataTest,1);

for i = 1:numTestSamples
    act(i) = YDataTest(i).getRow(1);
    result = modelShell.GetCurrentResult(XDataTest(i));
    predModel(i) = result(4);
    model1 = coreFunctions.Heating2DModel(30, 21, 21, 50, 1.5e-5, 0.3, 0.360, 30, 10);
    res1 = model1.CalcCoreFunction(XDataTest(i).getRow(1));
    model2 = coreFunctions.Heating2DModel(60, 21, 21, 60, 1.5e-5, 0.3, 0.360, res1, 10);
    res2 = model2.CalcCoreFunction(XDataTest(i).getRow(2));
    model3 = coreFunctions.Heating2DModel(90, 21, 21, 70, 1.5e-5, 0.3, 0.360, res2, 10);
    res3(i) = HeatBC_3.CalcCoreFunction(XDataTest(i).getRow(3));
end

%%
diagnostics_BW = plotHeatingPrediction(act,predModel);
diagnostics_3Heat = plotHeatingPrediction(act,res3);

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