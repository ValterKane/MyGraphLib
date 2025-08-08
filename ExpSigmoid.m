%% Генерация данных
import BWGraph.*;
import BWGraph.CustomMatrix.*;
import BWGraph.RandomGenerator.*;
import BWGraph.Trainer.*;
import coreFunctions.*;

rng(2);
% Общая функция для всех вершин
CoreF = SigmoidFunction();
LcoreF = LinearFunction();
alfaGen = AlphaGenerator(0.9);
betaGen = BetaGenerator(1);

nodeA = Node(1, 1, "White", CoreF);
nodeB = Node(2, 1, "Black", LcoreF);
nodeC = Node(3, 1, "Black", LcoreF);
nodeD = Node(4, 1, "Black", CoreF);
nodeE = Node(5, 1, "Black", LcoreF);
nodeF = Node(6, 1, "Black", LcoreF);
% nodeG = Node(7, 500, "Black", []);
% nodeH = Node(8, 500, "Black", []);

nodeD.addEdge(nodeB);
nodeC.addEdge(nodeB);
nodeE.addEdge(nodeB);
nodeF.addEdge(nodeB);
nodeB.addEdge(nodeA);


% Индивидуальные параметры для вершин (общие для всех экспериментов)
NodeSize = [1 1 1 1 1 1]; % Коэффициенты
NodeWeight = [1 1 1 1 1 1]; % Весовые коэффициенты вершин

% Создаем графовую модель
modelShell = GraphShell(alfaGen, betaGen,  nodeA, nodeB, nodeC, nodeD, nodeE, nodeF);

% Создаем входные данные
% Генерация данных с учетом индивидуальных характеристик вершин
numSamples = 500;
numOfNodes = numel(modelShell.ListOfNodes);
numOfWhiteNodes = modelShell.GetNumOfWhiteNode; % Получаем количество вершин
numInputParams = CoreF.GetNumOfInputParams(); % Получаем количество входных параметров
XData = repmat(BWMatrix(), numSamples, 1);
YData = repmat(BWMatrix(), numSamples, 1);

data_min = -10;
data_max = 10;

% Генерация и нормализация входных данных
for i = 1:numSamples
    % Генерируем случайное число в [-10, 10]
    raw_data = (data_max - data_min) * rand() + data_min;
    
    % Нормализуем в [0, 1]
    x = (raw_data - data_min) / (data_max - data_min);
    
    % Создаем вектор параметров
    xM = x * ones(numInputParams, 1);
    
    % Добавляем данные в XData
    for j = 1:numOfNodes
        XData(i) = XData(i).addRow(xM);
    end
end

% Генерация выходных данных (нормализованных)
for i = 1:numSamples
    yM = zeros(1, numOfWhiteNodes);
    for j = 1:numOfWhiteNodes
        % Получаем исходные данные (денормализация)
        original_data = XData(i).getRow(j) * (data_max - data_min) + data_min;
        
        % Вычисляем преобразование (например, квадрат)
        transformed_data = original_data ^ 2;
        
        % Нормализуем результат в [0, 1]
        % Если original_data ∈ [-10,10], то original_data^2 ∈ [0,100]
        yM(j) = (transformed_data - 0) / (100 - 0);
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

%% Диаграмма распределения
% Данные для диаграммы
trainCount = length(trainIndices);
testCount = length(testIndices);
sizes = [trainCount, testCount];

% Подписи с количеством данных
labels = {
    sprintf('Обучающая (%d)', trainCount), ...
    sprintf('Тестовая (%d)', testCount)
    };

% Создаем фигуру и настраиваем шрифт
figure;
set(gcf, 'DefaultTextFontName', 'Times New Roman');
set(gcf, 'DefaultAxesFontName', 'Times New Roman');
set(gcf, 'DefaultAxesFontSize', 14);

% Круговая диаграмма с выделением обучающей выборки
explode = [1 0]; % Выделяем обучающую выборку
h = pie(sizes, explode, labels);

% Устанавливаем красно-синюю цветовую схему
colors = [0.7 0.1 0.1; 0.1 0.3 0.7]; % [красный; синий]
colormap(colors);

% Настраиваем заголовок с увеличенным шрифтом
title(sprintf('Разбиение данных: %d (обуч) / %d (тест)', trainCount, testCount), ...
    'FontSize', 14, 'FontWeight', 'bold');

% Легенда с Times New Roman
legend(labels, 'Location', 'best', 'FontSize', 12);

%% Отобразить граф
modelShell.DrawGraph("Структура модели до настройки");

%% Настройка модели
% Задаем настройщик
trainer = Trainer(modelShell, 30);
% Вызываем настройщик
trainer.Train(XDataTrain, YDataTrain, XDataTest, YDataTest, 0.1, 0.1, 0.1, 1e-8, NodeSize, NodeWeight, 200, 1e12, -1e12, 1e-8);

%% Тестирование модели
whiteIdx = modelShell.GetWhiteNodesIndices;
data_min = 0;
data_max = 100;

numTestSamples = numel(YDataTest);
actualValue = zeros(1,numTestSamples);
predictionValue = zeros(1, numTestSamples);

for j = 1 : numTestSamples
    modelShell.Forward(XDataTest(j));
    actual = YDataTest(j).getRow(1) * (data_max - data_min) + data_min;
    predict = modelShell.GetModelResults() * (data_max - data_min) + data_min;
    predictionValue(j) = mean(predict(whiteIdx));
    actualValue(j) = mean(actual);
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
rng(123)
validSamples = 50;
data_min = -20;
data_max = 30;

XDataValid = repmat(BWMatrix(), validSamples, 1);
YDataValid = repmat(BWMatrix(), validSamples, 1);

for i = 1:validSamples
    xM = zeros(numInputParams,1);
    for j = 1:numOfNodes
        for k = 1:numInputParams
            data = randi([data_min,data_max]);
            x = (data + 20) / (30 + 20);
        end
        XDataValid(i) = XDataValid(i).addRow(xM);
    end
end

for i = 1:validSamples
    yM = zeros(1,numOfWhiteNodes);
     for j = 1:numOfWhiteNodes
        data = XDataValid(i).getRow(j) * (data_max - data_min) + data_max;
        yM(j) = (data ^ 2 - data_min) / (data_max - data_min);

    end
    YDataValid(i) = YDataValid(i).addRow(yM);
end

validValue = zeros(1,validSamples);
predictOnValid = zeros(1,validSamples);

for j = 1:validSamples
    modelShell.Forward(XDataValid(j));
    forecast = modelShell.GetModelResults() * (data_max - data_min) + data_min;
    actual = YDataValid(j).getRow(1) * (data_max - data_min) + data_min;
    validValue(j) = mean(actual);
    predictOnValid(j) = mean(forecast(whiteIdx));
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
fprintf('Результирующее MAE на валидации %2.3f\n', mae(predictOnValid,validValue))
fprintf('Результирующее MAPE на валидации %2.3f\n', mape(predictOnValid, validValue))

fprintf('Результирующее MAE на тесте %2.3f\n', mae(predictionValue,actualValue))
fprintf('Результирующее MAPE на тесте %2.3f\n', mape(predictionValue, actualValue))

%% Отобразить граф
modelShell.DrawGraph("Структура модели после настройки");