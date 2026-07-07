%% Очистить все
clear; clc;
rng(22);

import BWGraph.*;
import BWGraph.CustomMatrix.*;
import BWGraph.RandomGenerator.*;
import BWGraph.Trainer.*;

LM = coreFunctions.LinearFunction();
betaGen = FullRandomBetaGen(0,100); % Гиперпараметр

nodeA = Node(1, 1,'White',LM,'linear');
nodeB = Node(2, 1,'Black',LM,'linear');
nodeC = Node(3, 1,'Black',LM,'sigmoid');
nodeD = Node(4, 1,'Black',LM,'sigmoid');

nodeD.addEdge(nodeB);
nodeC.addEdge(nodeB);
nodeB.addEdge(nodeA);

% Гиперпараметр
NodeWeight = [1 1 1 1]; % Весовые коэффициенты вершин

modelShell = GraphShell(betaGen,NodeWeight, nodeA, nodeB, nodeC, nodeD);

%% Создаем входные данные
% Генерация данных с учетом индивидуальных характеристик вершин
numSamples = 500;
numOfNodes = numel(modelShell.ListOfNodes);
numOfWhiteNodes = modelShell.GetNumOfWhiteNode; % Получаем количество вершин
numInputParams = LM.GetNumOfInputParams(); % Получаем количество входных параметров
XData = repmat(BWMatrix(), numSamples, 1);
YData = repmat(BWMatrix(), numSamples, 1);

for i = 1:numSamples 
    xM = zeros(numInputParams,1);
    x = randi([1,100]);
    for j = 1:numOfNodes
        for k = 1:numInputParams
            xM(k) = x;
        end
        XData(i) = XData(i).addRow(xM);
    end 
end

for i = 1:numSamples
    yM = zeros(1,numOfWhiteNodes);
    for j = 1:numOfWhiteNodes
         yM(j) = 10*x+12;
    end
    YData(i) = YData(i).addRow(yM);
end

% Делим выборку на подвыборки 80(обуч)% / 20(тест)%
indices = randperm(numSamples);
splitPoint = round(0.8 * numSamples);
trainIndices = indices(1:splitPoint);
testIndices = indices(splitPoint+1:end);
% Определим обучающую и тестовую выборку
XDataTrain = XData(trainIndices);
YDataTrain = YData(trainIndices);
XDataTest = XData(testIndices);
YDataTest = YData(testIndices);

%% Настройка учителя
% Опции настройки
trainerOptions = TrainingOptions( ...
    "LearningRate", 0.01, ...
    "NodeSize", [1 1 1 1 1], ...
    "LRDecayInterval", 50, ...
    "TargetError", 0.5, ...
    "Epoches", 500, ...
    "AlphaMin", 0.01, ...
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