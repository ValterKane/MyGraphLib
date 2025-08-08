% Тест на простых данных x1+x2 = y
%% Генерация данных
import BWGraph.*;
import BWGraph.CustomMatrix.*;
import BWGraph.RandomGenerator.*;
import BWGraph.Trainer.*;
import coreFunctions.*;

rng(2);

% Общая функция для всех вершин
CoreF = SimpleAddingCoreFunction(2);
alfaGen = AlphaGenerator(0.1);
betaGen = BetaGenerator(0.1);

% Создаем вершины
node1 = Node(1, 5,'White',CoreF);

% Определяем связи между вершинами
node1.addEdge(node1);

% Индивидуальные параметры для вершин (общие для всех экспериментов)
NodeSize = 1.0; % Коэффициенты 
NodeWeight = 1.0; % Весовые коэффициенты вершин

% Создаем графовую модель
modelShell = GraphShell(alfaGen, betaGen, node1);

% Создаем входные данные
% Генерация данных с учетом индивидуальных характеристик вершин
numSamples = 500;
numOfNodes = numel(modelShell.ListOfNodes);
numOfWhiteNodes = modelShell.GetNumOfWhiteNode; % Получаем количество вершин
numInputParams = CoreF.GetNumOfInputParams(); % Получаем количество входных параметров

XData = repmat(BWMatrix(), numSamples, 1);
YData = repmat(BWMatrix(), numSamples, 1);

for i = 1:numSamples 
    xM = zeros(numInputParams,1);
    for j = 1:numOfNodes
        for k = 1:numInputParams
            xM(k) = randi([-100,100]);
        end
        XData(i) = XData(i).addRow(xM);
    end 
end

for i = 1:numSamples
    yM = zeros(1,numOfWhiteNodes);
    for j = 1:numOfWhiteNodes
         yM(j) = sum(XData(i).getRow(j));
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

%% Настройка модели
% Задаем настройщик
trainer = Trainer(modelShell, 10);
% Вызываем настройщик
trainer.Train(XDataTrain, YDataTrain, XDataTest, YDataTest, 0.1, 0.9, 0.99,1e-8,NodeSize,NodeWeight,5000, 1e12, -1e12);

%% Тестирование модели
modelShell.Forward(XDataTest(1));
numTestSamples = numel(YDataTest);
actualValue = zeros(1,numTestSamples);
predictionValue = zeros(1, numTestSamples);

for j = 1 : numTestSamples
    modelShell.Forward(XDataTest(j));
    actual = YDataTest(j).getRow(1);
    predict = modelShell.GetModelResults();
    predictionValue(j) = predict(1);
    actualValue(j) = actual(1);
end

% Строим график сравнения
figure;
hold on;
grid on;

% Рисуем линии фактических и модельных значений
plot(1:numTestSamples, actualValue, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Фактические значения');
plot(1:numTestSamples, predictionValue, 'r--s', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Модельные значения');

% Настраиваем график
xlabel('Номер тестового примера');
ylabel('Значение');
title('Сравнение модельных и фактических значений на тестовой выборке');
legend('show', 'Location', 'best');

% Добавляем линию нуля (если нужно)
% line([1 numTestSamples], [0 0], 'Color', 'k', 'LineStyle', '--');

hold off;

%% Валидация
validSamples = 20;
rng(123);
XDataValid = repmat(BWMatrix(), validSamples, 1);
YDataValid = repmat(BWMatrix(), validSamples, 1);

for i = 1:validSamples 
    xM = zeros(numInputParams,1);
    for j = 1:numOfNodes
        for k = 1:numInputParams
            xM(k) = randi([-720,1000]);
        end
        XDataValid(i) = XDataValid(i).addRow(xM);
    end 
end

for i = 1:validSamples
    yM = zeros(1,numOfWhiteNodes);
    for j = 1:numOfWhiteNodes
         yM(j) = sum(XDataValid(i).getRow(j));
    end
    YDataValid(i) = YDataValid(i).addRow(yM);
end

validValue = zeros(1,validSamples);
predictOnValid = zeros(1, validSamples);

for j = 1:validSamples
    modelShell.Forward(XDataValid(j));
    forecast = modelShell.GetModelResults();
    validValue(j) = YDataValid(j).getRow(1);
    predictOnValid(j) = forecast;
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

%% Показатели и метрики
fprintf('Результирующее MAE %2.3f\n', mae(validValue,predictOnValid))
fprintf('Результирующее MAE %2.3f\n', mape(validValue,predictOnValid))

%% Сохранить модель в файл
modelShell.SaveToFile('C:\Users\darkd\Desktop\2024-2025\Графовая модель\Эксперименты\х1+х2=y\Модель\model');

%% Загрузить модель из файла
