%% Очистить все
clear; clc; close all;
rng(1111);
%% Подготовка
import BWGraph.*;
import BWGraph.CustomMatrix.*;
import BWGraph.RandomGenerator.*;
import BWGraph.Trainer.*;

HeatBC_1 = coreFunctions.Heating2DModel(30, 21, 21, 50, 1.5e-5, 0.3, 0.360, 30, 10);
HeatBC_2 = coreFunctions.Heating2DModel(60, 21, 21, 60, 1.5e-5, 0.3, 0.360, 500, 10);
HeatBC_3 = coreFunctions.Heating2DModel(90, 21, 21, 70, 1.5e-5, 0.3, 0.360, 1000, 10);

LinearTemper = coreFunctions.LinearFunction();

alfaGen = FullRandomAlfaGen(1,1e2);
betaGen = FullRandomBetaGen(1,1e4);

%% Создание и настройка модели
nodeA = Node(1, 30,'Black',HeatBC_1);
nodeB = Node(2, 30,'Black',HeatBC_2);
nodeC = Node(3, 30,'Black',LinearTemper);
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
NodeSize = [1.3 1.3 1.3 1.3 1.5]; % Коэффициенты 
NodeWeight = [2 2 2 2 3]; % Весовые коэффициенты вершин

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

%% Загрузка других данных 
if ~exist("data", 'var')
    data = readtable("C:\Users\darkd\Desktop\2024-2025\Математическая модель многозонной печи\Регрессия\DataForCompr.xlsx");
end

numOfNodes = numel(modelShell.ListOfNodes);
numSamples = size(data,1);
numOfWhiteNodes = modelShell.GetNumOfWhiteNode; 
numInputParams = HeatBC_1.GetNumOfInputParams();
totalBatch = 300;

% Определим матрицы входа и выхода
XData = repmat(BWMatrix(), numSamples, 1);
YData = repmat(BWMatrix(), numSamples, 1);

t1 = table2array(data(1:totalBatch,"F12_TimeDiff"));
t2 = table2array(data(1:totalBatch,"F34_TimeDiff"));
t3 = table2array(data(1:totalBatch,"F56_TimeDiff"));

T1 = (data{1:totalBatch,'F12_TL'} + data{1:totalBatch, 'F12_TR'})/2;
T2 = (data{1:totalBatch,'F34_TL'} + data{1:totalBatch, 'F34_TR'})/2;
T3 = (data{1:totalBatch,'F56_TL'} + data{1:totalBatch, 'F56_TR'})/2;
T_y = data{1:totalBatch,'Tmax'}; 

data_for_one = [t1, T1];
data_for_two = [t2, T2];
data_for_three = [t3, T3];


for i = 1:totalBatch
    XData(i) = XData(i).addRow(data_for_one(i,:));
    XData(i) = XData(i).addRow(data_for_two(i,:));
    XData(i) = XData(i).addRow(data_for_one(i,:));
    % XData(i) = XData(i).addRow(data_for_two(i,:));
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


%% Загрузка новых данных со скользящим средним
if ~exist("data", 'var')
    data = readtable("C:\Users\darkd\Desktop\2024-2025\Математическая модель многозонной печи\Регрессия\DataForCompr.xlsx");
end

% Параметры скользящего среднего
windowSize = 10;     % Размер окна
stepSize = 1;       % Шаг выборки

totalBatch = 300;

% Функция для применения скользящего среднего с шагом
function smoothed = smoothAndSample(data, windowSize, stepSize)
    % Используем встроенную функцию movmean для скользящего среднего
    smoothed_full = movmean(data, windowSize, 'Endpoints', 'shrink');
    
    % Выборка с заданным шагом
    smoothed = smoothed_full(1:stepSize:end);
end

% Получение исходных данных
t1_raw = table2array(data(1:totalBatch,"F12_TimeDiff"));
t2_raw = table2array(data(1:totalBatch,"F34_TimeDiff"));
t3_raw = table2array(data(1:totalBatch,"F56_TimeDiff"));

T1_raw = (data{1:totalBatch,'F12_TL'} + data{1:totalBatch, 'F12_TR'})/2;
T2_raw = (data{1:totalBatch,'F34_TL'} + data{1:totalBatch, 'F34_TR'})/2;
T3_raw = (data{1:totalBatch,'F56_TL'} + data{1:totalBatch, 'F56_TR'})/2;
T_y_raw = data{1:totalBatch,'Tmax'};

% Применение скользящего среднего с шагом
t1 = smoothAndSample(t1_raw, windowSize, stepSize);
t2 = smoothAndSample(t2_raw, windowSize, stepSize);
t3 = smoothAndSample(t3_raw, windowSize, stepSize);
T1 = smoothAndSample(T1_raw, windowSize, stepSize);
T2 = smoothAndSample(T2_raw, windowSize, stepSize);
T3 = smoothAndSample(T3_raw, windowSize, stepSize);
T_y = smoothAndSample(T_y_raw, windowSize, stepSize);

% Обновление totalBatch до минимальной длины всех массивов
minLength = min([length(t1), length(t2), length(t3), ...
                 length(T1), length(T2), length(T3), length(T_y)]);
totalBatch = minLength;

% Обрезка всех массивов до одинаковой длины
t1 = t1(1:totalBatch);
t2 = t2(1:totalBatch);
t3 = t3(1:totalBatch);
T1 = T1(1:totalBatch);
T2 = T2(1:totalBatch);
T3 = T3(1:totalBatch);
T_y = T_y(1:totalBatch);

fprintf('Новый размер выборки: %d (windowSize=%d, stepSize=%d)\n', totalBatch, windowSize, stepSize);

% Далее оригинальный код продолжается без изменений...
data_for_one = [t1, T1];
data_for_two = [t2, T2];
data_for_three = [t3, T3];

% Определим матрицы входа и выхода
XData = repmat(BWMatrix(), totalBatch, 1);
YData = repmat(BWMatrix(), totalBatch, 1);

for i = 1:totalBatch
    XData(i) = XData(i).addRow(data_for_one(i,:));
    XData(i) = XData(i).addRow(data_for_two(i,:));
    XData(i) = XData(i).addRow(T1(i,:));
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
%% Создаем настройщик
trainer = Trainer(modelShell, 1);
%% Настройка
trainer.Train(XDataTrain, YDataTrain, XDataTest, YDataTest, 0.1, 0.9, 0.999,1e-8, NodeSize, NodeWeight, 1000, 1e7, -1e7, 20, 0.1, 0.001, [], 'mae');
 %%
test = HeatBC_3.CalcCoreFunction(XData(1,1).getRow(3)) - 273;
modelShell.Forward(XData(1,1));
test2 = modelShell.GetModelResults - 273;

absolute_erro = test2-test;

%% Тестирование

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
plot(1:numTestSamples, predModel, 'r--s', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Значение BW-модели');
plot(1:numTestSamples, res3, 'g-s', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Значение по 3-Heat модели');

% Настраиваем график
xlabel('Номер тестового примера');
ylabel('Значение');
title('Апробация модели на тестовом подмножестве');
legend('show', 'Location', 'best');

set(gca, 'FontSize', 14, 'FontWeight', 'bold');

grid on;
hold off;

mae_1 = sum(abs(act - predModel)) / numel(act);
fprintf("МАЕ по модели графа = %.2f\n", mae_1);
mae_2 = sum(abs(act - res3)) / numel(act);
fprintf("МАЕ по трем последовательным моделям = %.2f\n", mae_2);

fprintf("Среднее по BW-модели = %.2f\n", mean(predModel));
fprintf("Среднее по Heat-модели = %.2f\n", mean(res3));
fprintf("Среднее по исходным данным = %.2f\n", mean(act));


fprintf("R2 по BW-модели = %.2f\n", calculateR2(act,predModel));
fprintf("R2 по 3-Heat-модели = %.2f\n", calculateR2(act,res3));

function R2 = calculateR2(y_true, y_pred)
    % y_true - реальные значения
    % y_pred - предсказанные значения
    
    % Проверка размеров
    if length(y_true) ~= length(y_pred)
        error('Размеры векторов должны совпадать');
    end
    
    % Расчет суммы квадратов остатков (SS_res)
    SS_res = sum((y_true - y_pred).^2);
    
    % Расчет общей суммы квадратов (SS_tot)
    SS_tot = sum((y_true - mean(y_true)).^2);
    
    % Расчет R²
    R2 = 1 - SS_res/SS_tot;
end
%% Загрузка данных валидации
if ~exist("validData", 'var')
    validData = readtable("C:\Users\darkd\Desktop\2024-2025\Математическая модель многозонной печи\Регрессия\DataForCompr.xlsx");
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