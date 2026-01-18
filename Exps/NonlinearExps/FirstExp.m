%% Очистить все
clear; clc; close all;
rng(111); 

%% Подготовка
import BWGraph.*;
import BWGraph.CustomMatrix.*;
import BWGraph.RandomGenerator.*;
import BWGraph.Trainer.*;
import BWGraph.NonLinearBWGraph.*;
import BWGraph.NonLinearBWGraph.Trainer.*;


LinearRegression = coreFunctions.HeatLinearRegression();

alfaGen = FullRandomGen(1,10,'Alfa');
betaGen = FullRandomGen(1,1e4,'Beta');
gamaGen = FullRandomGen(1,10,'Gamma');
deltaGen = FullRandomGen(1,10,'Delta');

%% Создание и настройка модели
nodeA = Node(1, 1,'Black',LinearRegression);
nodeB = Node(2, 1,'Black',LinearRegression);
% nodeC = Node(3, 1,'Black',LinearRegression);
% nodeD = Node(4, 1,'Black',LinearRegression);
nodeE = Node(3, 1,'White',LinearRegression);

nodeE.addEdge(nodeA);
nodeE.addEdge(nodeB);
% nodeE.addEdge(nodeC);
% nodeE.addEdge(nodeD);

nodeA.addEdge(nodeE);
nodeB.addEdge(nodeE);
% nodeC.addEdge(nodeE);
% nodeD.addEdge(nodeE);

% Создаем графовую модель
modelShell = GraphShellNonlinear(alfaGen,betaGen,gamaGen,deltaGen,nodeA,nodeB, nodeE);

% Индивидуальные параметры для вершин (общие для всех экспериментов)
NodeSize = [1 1 1]; % Коэффициенты 
NodeWeight = [1 1 1]; % Весовые коэффициенты вершин

%% Отрисовать граф
modelShell.DrawGraph('Модель нагрева');

%% Загрузка данных
if ~exist("data", 'var')
    data = readtable("C:\Users\darkd\Desktop\2024-2025\Математическая модель многозонной печи\Регрессия\Data.xlsx");
end

numOfNodes = numel(modelShell.ListOfNodes);
numOfWhiteNodes = modelShell.numOfWhiteNodes; % Получаем количество вершин
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
    % XData(i) = XData(i).addRow(X(i,:));
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
trainer = NonLinearTrainer(modelShell, 1);
%% Настройка
trainer.Train(XDataTrain, YDataTrain, XDataTest, YDataTest, 0.1, 0.9, 0.9,1e-8, NodeSize, NodeWeight, 1000, 1e8, -1e8, 4.0, 0.5, 0.001, [], 'mae');

%% Очистить все
clear; clc; close all;
rng(1111);
%% Подготовка
import BWGraph.*;
import BWGraph.CustomMatrix.*;
import BWGraph.RandomGenerator.*;
import BWGraph.Trainer.*;
import BWGraph.NonLinearBWGraph.*;
import BWGraph.NonLinearBWGraph.Trainer.*;

HeatBC_1 = coreFunctions.Heating2DModel(30, 21, 21, 50, 1.5e-5, 0.3, 0.360, 30, 10);
HeatBC_2 = coreFunctions.Heating2DModel(60, 21, 21, 60, 1.5e-5, 0.3, 0.360, 700, 10);
HeatBC_3 = coreFunctions.Heating2DModel(90, 21, 21, 70, 1.5e-5, 0.3, 0.360, 900, 10);

alfaGen = FullRandomGen(1,10,'Alfa');
betaGen = FullRandomGen(1,1e4,'Beta');
gamaGen = FullRandomGen(0.1,1,'Gamma');
deltaGen = FullRandomGen(0.1,1,'Delta');

%% Создание и настройка модели
nodeA = Node(1, 30,'Black',HeatBC_1);
nodeB = Node(3, 30,'Black',HeatBC_2);
nodeC = Node(5, 30,'White',HeatBC_3);

nodeA.addEdge(nodeB);
nodeB.addEdge(nodeC);

nodeB.addEdge(nodeA);
nodeC.addEdge(nodeB);

% Создаем графовую модель
modelShell = GraphShellNonlinear(alfaGen,betaGen, gamaGen, deltaGen, nodeA,nodeB,nodeC);

% Индивидуальные параметры для вершин (общие для всех экспериментов)
NodeSize = [1 1 1.5]; % Коэффициенты 
NodeWeight = [1.2 1.2 1.5]; % Весовые коэффициенты вершин

%% Отрисовать граф
modelShell.DrawGraph('Модель нагрева');

%% Загрузка реальных данных
data = readtable('C:\Users\darkd\Desktop\2024-2025\Математическая модель многозонной печи\Готовые данные по первой садке.xlsx');
numOfNodes = numel(modelShell.ListOfNodes);
numOfWhiteNodes = modelShell.numOfWhiteNodes; % Получаем количество вершин
numInputParams = HeatBC_1.GetNumOfInputParams(); % Получаем количество входных параметров
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
splitPoint = round(0.8 * numSamples);
trainIndices = indices(1:splitPoint);
testIndices = indices(splitPoint+1:end);
 
% Определим обучающую и тестовую выборку
XDataTrain = XData(trainIndices);
YDataTrain = YData(trainIndices);
XDataTest = XData(testIndices);
YDataTest = YData(testIndices);

%% Создаем настройщик
trainer = NonLinearTrainer(modelShell, 1);
%% Настройка
trainer.Train(XDataTrain, YDataTrain, XDataTest, YDataTest, 0.1, 0.9, 0.9,1e-8, NodeSize, NodeWeight, 1000, 1e8, -1e8, 4.0, 0.5, 0.001, [], 'mae');