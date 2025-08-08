%% Подготовка
import BWGraph.*;
import BWGraph.CustomMatrix.*;
import BWGraph.RandomGenerator.*;
import BWGraph.Trainer.*;
rng(2);

%% Общая функция для всех вершин
HeatBC = coreFunctions.HeatTransferBC(320.0, 0.72);
alfaGen = AlphaGenerator(0.2);
betaGen = BetaGenerator(0.9);

%% Создаем вершины
nodeA = Node(1, 10,'White',HeatBC);
nodeB = Node(2, 10,'White',HeatBC);
nodeC = Node(3, 10,'White',HeatBC);
nodeD = Node(4, 10,'Black',[]);

% Добавляем соседей
% Вершина А
nodeA.addEdge(nodeB);
nodeA.addEdge(nodeC);
nodeA.addEdge(nodeD);
% Вершина B
nodeB.addEdge(nodeA);
nodeB.addEdge(nodeC);
nodeB.addEdge(nodeD);
% Вершина С
nodeC.addEdge(nodeA);
nodeC.addEdge(nodeB);
nodeC.addEdge(nodeD);
% Вершина D
nodeD.addEdge(nodeA);
nodeD.addEdge(nodeC);
nodeD.addEdge(nodeB);

% Создаем графовую модель
modelShell = GraphShell(alfaGen,betaGen,nodeA,nodeB,nodeC,nodeD);

%% Создаем входные данные
% Генерация данных с учетом индивидуальных характеристик вершин
numSamples = 500;
numOfNodes = numel(modelShell.ListOfNodes);
numOfWhiteNodes = modelShell.GetNumOfWhiteNode; % Получаем количество вершин
numInputParams = HeatBC.GetNumOfInputParams(); % Получаем количество входных параметров

XData = repmat(BWMatrix(), numSamples, 1);
YData = repmat(BWMatrix(), numSamples, 1);

for i = 1:numSamples 
    for j = 1:numOfNodes
        % Генерация данных для текущего примера
        target = 1100 + randi([0,10]);
        infTemp = 1050 + randi([0,15]);
        surfTemp = 1200 + randi([0,20]);
        XData(i) = XData(i).addRow([target;infTemp;surfTemp]);
    end
end

for i = 1:numSamples
    yMatrix = zeros(1,numOfWhiteNodes);
    for j = 1:numOfWhiteNodes
        yValue = XData(i).getRow(j);
        yMatrix(j) = yValue(1);
    end
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

% Индивидуальные параметры для вершин
NodeSize = [1.0, 1.0, 1.0, .5]; % Коэффициенты 
NodeWeight = [1.0, 1.0, 1.0, 0.5]; % Весовые коэффициенты вершин

%% Отобразить граф
modelShell.DrawGraph("Структура модели после настройки");

%% Настройка
trainer = Trainer(modelShell, 10);

trainer.Train(XDataTrain, YDataTrain, XDataTest, YDataTest, 0.1, 0.3, 0.5,1e-8, NodeSize, NodeWeight, 100, 1e12, -1e12, 5.0, 1.0, 0.5, [], 'mae');
%% Тестирование
numTestSamples = numel(YDataTest);
actualValue = zeros(1,numTestSamples);
predictionValue = zeros(1, numTestSamples);
index = 1;

for j = 1:numTestSamples
    modelShell.Forward(XDataTest(j));
    actual = YDataTest(j).getRow(1);
    predict = modelShell.GetModelResults();
    actualValue(j) = mean(actual);
    predictionValue(j) = mean(predict(modelShell.GetWhiteNodesIndices));
end

% Строим график сравнения
figure;
hold on;
grid on;

% Рисуем линии фактических и модельных значений
plot(1:numTestSamples, actualValue, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Фактические значения');
plot(1:numTestSamples, predictionValue, 'r-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Модельные значения');

% Настраиваем график
xlabel('Номер тестового примера');
ylabel('Значение');
title('Сравнение модельных и фактических значений на тестовой выборке');
legend('show', 'Location', 'best');

% Добавляем линию нуля (если нужно)
% line([1 numTestSamples], [0 0], 'Color', 'k', 'LineStyle', '--');

hold off;

%% Аналитические оценки

fprintf('Результирующее MAE %2.3f\n', mae(actualValue,predictionValue))

%% Валидация
validSamples = 100;

XDataValid = repmat(BWMatrix(), validSamples, 1);
YDataValid = repmat(BWMatrix(), validSamples, 1);

for i = 1:validSamples 
    for j = 1:numOfNodes
        % Генерация данных для текущего примера
        target = 1100 + randi([0,10]);
        infTemp = 1050 + randi([0,15]);
        surfTemp = 1200 + randi([0,20]);
        XDataValid(i) = XDataValid(i).addRow([target;infTemp;surfTemp]);
    end
end

for i = 1:validSamples
    yMatrix = zeros(1,numOfWhiteNodes);
    for j = 1:numOfWhiteNodes
        yValue = XDataValid(i).getRow(j);
        yMatrix(j) = yValue(1);
    end
    YDataValid(i) = YDataValid(i).addRow(yMatrix);
end

validValue = zeros(1,validSamples);
predictOnValid = zeros(1, validSamples);

for j = 1:validSamples
    modelShell.Forward(XDataValid(j));
    forecast = modelShell.GetModelResults();
    validValue(j) = mean(YDataValid(j).getRow(1));
    predictOnValid(j) = mean(forecast(modelShell.GetWhiteNodesIndices));
end

% Строим график сравнения
figure;
hold on;
grid on;

% Рисуем линии фактических и модельных значений
plot(1:validSamples, validValue, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Фактические значения');
plot(1:validSamples, predictOnValid, 'r-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Модельные значения');

% Настраиваем график
xlabel('Номер тестового примера');
ylabel('Значение');
title('Сравнение модельных и фактических значений на тестовой выборке');
legend('show', 'Location', 'best');

% Добавляем линию нуля (если нужно)
% line([1 numTestSamples], [0 0], 'Color', 'k', 'LineStyle', '--');

hold off;

%% Анализ метрик
fprintf('______________________________________\n')
fprintf('Результирующее MAE на валидации %2.3f\n', mae(predictOnValid,validValue))
fprintf('Результирующее MAPE на валидации %2.3f\n', mape(predictOnValid, validValue))
fprintf('######################################\n')
fprintf('Результирующее MAE на тесте %2.3f\n', mae(predictionValue,actualValue))
fprintf('Результирующее MAPE на тесте %2.3f\n', mape(predictionValue, actualValue))
fprintf('######################################\n')
fprintf('MO прогноза на валидационном множестве: %3f\n', mean(predictOnValid))
fprintf('MO валидационного множества: %3f\n', mean(validValue))
fprintf('MO прогноза на тестовом множестве: %3f\n', mean(predictionValue))
fprintf('MO тестового множества: %3f\n', mean(actualValue))
fprintf('######################################\n')
fprintf('VAR прогноза на валидационном множестве: %3f\n', var(predictOnValid))
fprintf('VAR валидационного множества: %3f\n', var(validValue))
fprintf('VAR прогноза на тестовом множестве: %3f\n', var(predictionValue))
fprintf('VAR тестового множества: %3f\n', var(actualValue))
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
