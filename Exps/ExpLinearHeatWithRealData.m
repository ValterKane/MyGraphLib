%% Очистить все
clear; clc; close all;
rng(115); 

%% Подготовка
import BWGraph.*;
import BWGraph.CustomMatrix.*;
import BWGraph.RandomGenerator.*;
import BWGraph.Trainer.*;

LinearRegression = coreFunctions.HeatLinearRegression();

alfaGen = FullRandomAlfaGen(-100, 100);
betaGen = FullRandomBetaGen(1, 1e4);

%% Создание и настройка модели
nodeA = Node(1, 1,'Black',LinearRegression);
nodeB = Node(2, 1,'Black',LinearRegression);
nodeC = Node(3, 1,'Black',LinearRegression);
% nodeD = Node(4, 1,'Black',LinearRegression);
nodeE = Node(4, 1,'White',LinearRegression);

nodeE.addEdge(nodeA);
nodeE.addEdge(nodeB);
nodeE.addEdge(nodeC);
% nodeE.addEdge(nodeD);

nodeA.addEdge(nodeE);
nodeB.addEdge(nodeE);
nodeC.addEdge(nodeE);
% nodeD.addEdge(nodeE);

% Создаем графовую модель
modelShell = GraphShell(alfaGen,betaGen,nodeA,nodeB,nodeC, nodeE);

% Индивидуальные параметры для вершин (общие для всех экспериментов)
NodeSize = [1 1 1 1]; % Коэффициенты 
NodeWeight = [1 1 1 1]; % Весовые коэффициенты вершин

%% Отрисовать граф
modelShell.DrawGraph('Модель нагрева');
%% Загрузка других данных 
if ~exist("data", 'var')
    data = readtable("C:\Users\darkd\Desktop\2024-2025\Математическая модель многозонной печи\Регрессия\DataForCompr.xlsx");
end

numOfNodes = numel(modelShell.ListOfNodes);
numOfWhiteNodes = modelShell.GetNumOfWhiteNode; % Получаем количество вершин
numInputParams = LinearRegression.GetNumOfInputParams(); % Получаем количество входных параметров
numSamples = 131;

% Определим матрицы входа и выхода
XData = repmat(BWMatrix(), numSamples, 1);
YData = repmat(BWMatrix(), numSamples, 1);

t1 = table2array(data(1:131,"F12_TimeDiff"));
t2 = table2array(data(1:131,"F34_TimeDiff"));
t3 = table2array(data(1:131,"F56_TimeDiff"));
tc = table2array(data(1:131,"F_AF"));
T1 = (data{1:131,'F12_TL'} + data{1:131, 'F12_TR'})/2;
T2 = (data{1:131,'F34_TL'} + data{1:131, 'F34_TR'})/2;
T3 = (data{1:131,'F56_TL'} + data{1:131, 'F56_TR'})/2;
T_y = data{1:131,'Tmax'}; 

X = [t1,t2,t3,T1,T2,T3,tc];
Y = T_y;

for i = 1:numSamples
    XData(i) = XData(i).addRow(X(i,:));
    XData(i) = XData(i).addRow(X(i,:));
    XData(i) = XData(i).addRow(X(i,:));
    XData(i) = XData(i).addRow(X(i,:));
end

for i = 1:numSamples
    YData(i) = YData(i).addRow(Y(i,:));
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

%% Загрузка данных
if ~exist("data", 'var')
    data = readtable("C:\Users\darkd\Desktop\2024-2025\Математическая модель многозонной печи\Регрессия\Data.xlsx");
end

numOfNodes = numel(modelShell.ListOfNodes);
numOfWhiteNodes = modelShell.GetNumOfWhiteNode; % Получаем количество вершин
numInputParams = LinearRegression.GetNumOfInputParams(); % Получаем количество входных параметров
numSamples = height(data);

% Определим матрицы входа и выхода
XData = repmat(BWMatrix(), numSamples, 1);
YData = repmat(BWMatrix(), numSamples, 1);

t1 = (data{:,'H12'} * 60 + data{:,'M12'})*60;
t2 = (data{:,'H34'} * 60 + data{:,'M34'})*60;
t3 = (data{:,'H56'} * 60 + data{:,'M56'})*60;
T1 = (data{:,'minT12'} + data{:, 'maxT12'})/2;
T2 = (data{:,'minT34'} + data{:, 'maxT34'})/2;
T3 = (data{:,'minT56'} + data{:, 'maxT56'})/2;
tc = seconds(datetime(data{:,'stanTimeStamp'},'ConvertFrom','excel') - datetime(data{:,'duoTimeStamp'}, 'ConvertFrom', 'excel'));
% T_y = (data{:,'maxTdou'} + data{:, 'minTduo'})/2;
T_y = data{:,'maxTstan'}; 

X = [t1,t2,t3,T1,T2,T3,tc];
Y = T_y;

for i = 1:numSamples
    % XData(i) = XData(i).addRow(X(i,:));
    XData(i) = XData(i).addRow(X(i,:));
    XData(i) = XData(i).addRow(X(i,:));
    XData(i) = XData(i).addRow(X(i,:));
    XData(i) = XData(i).addRow(X(i,:));
end

for i = 1:numSamples
    YData(i) = YData(i).addRow(Y(i,:));
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

%% Создаем настройщик
trainer = Trainer(modelShell, 1);
%% Настройка
trainer.Train(XDataTrain, YDataTrain, XDataTest, YDataTest, 0.1, 0.9, 0.9,1e-8, NodeSize, NodeWeight, 1000, 1e8, -1e8, 4.0, 0.5, 0.001, [], 'mae');
%% Ручная проверка

numTestSamples = size(XDataTest,1);

for i = 1:numTestSamples
    data = XDataTest(i).getRow(1);
    act(1,i) = YDataTest(i).getRow(1);
    pred(1,i) = LinearRegression.CalcCoreFunction(data);
    result = modelShell.GetCurrentResult(XDataTest(i));
    predModel(1,i) = result(4);
end

mae_predModel = sum(abs(act - predModel)) / numel(act);
mae_pred = sum(abs(act - pred)) / numel(act);

fprintf("МАЕ по линейной модели = %.2f\n", mae_pred);
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
plot(1:numTestSamples, pred, 'g--s', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Значения линейной регрессии');

% Настраиваем график
xlabel('Номер тестового примера');
ylabel('Значение');
title('Апробация модели на тестовом подмножестве');
legend('show', 'Location', 'best');

set(gca, 'FontSize', 14, 'FontWeight', 'bold');

grid on;
hold off;

%% Загрузка данных валидации
if ~exist("validData", 'var')
    validData = readtable("C:\Users\darkd\Desktop\2024-2025\Математическая модель многозонной печи\Регрессия\DataForCompr.xlsx");
end

numOfNodes = numel(modelShell.ListOfNodes);
numOfWhiteNodes = modelShell.GetNumOfWhiteNode; % Получаем количество вершин
numInputParams = LinearRegression.GetNumOfInputParams(); % Получаем количество входных параметров
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

X = [t1,t2,t3,T1,T2,T3,tc];
Y = T_y;

for i = 1:numSamples
    XValidData(i) = XValidData(i).addRow(X(i,:));
    XValidData(i) = XValidData(i).addRow(X(i,:));
    XValidData(i) = XValidData(i).addRow(X(i,:));
    XValidData(i) = XValidData(i).addRow(X(i,:));
end

for i = 1:numSamples
    YValidData(i) = YValidData(i).addRow(Y(i,:));
end

%% Проверка на валидации

numTestSamples = size(XValidData,1);

for i = 1:numTestSamples
    data = XValidData(i).getRow(3);
    act(1,i) = YValidData(i).getRow(1);
    pred(1,i) = LinearRegression.CalcCoreFunction(data);
    result = modelShell.GetCurrentResult(XValidData(i));
    predModel(1,i) = result(4);
end

mae_predModel = sum(abs(act - predModel)) / numel(act);
mae_pred = sum(abs(act - pred)) / numel(act);

fprintf("МАЕ по линейной модели = %.2f\n", mae_pred);
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
plot(1:numTestSamples, pred, 'g--s', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Значения линейной регрессии');

% Настраиваем график
xlabel('Номер тестового примера');
ylabel('Значение');
title('Апробация модели на тестовом подмножестве');
legend('show', 'Location', 'best');

set(gca, 'FontSize', 14, 'FontWeight', 'bold');

grid on;
hold off;
