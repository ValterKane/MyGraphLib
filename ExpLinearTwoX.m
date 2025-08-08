%% Генерация данных
import BWGraph.*;
import BWGraph.CustomMatrix.*;
import BWGraph.RandomGenerator.*;
import BWGraph.Trainer.*;
import coreFunctions.*;

rng(2);
% Общая функция для всех вершин
CoreF = LinearRegression(2);
alfaGen = AlphaGenerator(0.9);
betaGen = BetaGenerator(1);

nodeA = Node(1, 1, "White", CoreF);
nodeB = Node(2, 1, "Black",  CoreF);
nodeC = Node(3, 1, "Black",  CoreF);
nodeD = Node(4, 1, "Black",  CoreF);
nodeE = Node(5, 1, "Black", []);
nodeF = Node(6, 1, "Black", []);
% nodeG = Node(7, 1, "Black", CoreF);
% nodeH = Node(8, 1, "Black", CoreF);

nodeE.addEdge(nodeA);
nodeF.addEdge(nodeA);
nodeA.addEdge(nodeB);
nodeA.addEdge(nodeC);
nodeA.addEdge(nodeD);

% nodeE.addEdge(nodeA);
% nodeF.addEdge(nodeA);
% % nodeA.addEdge(nodeG);
% % nodeA.addEdge(nodeH);


% Индивидуальные параметры для вершин (общие для всех экспериментов)
NodeSize = [1 1 1 1 1 1 1 1]; % Коэффициенты 
NodeWeight = [1 1 1 1 1 1 1 1]; % Весовые коэффициенты вершин

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


for i = 1:numSamples
    % Генерация Х
    xM = zeros(numInputParams,1);
    for k = 1:numInputParams
        xM(k) = randi([1,20]);
    end
    XData(i) = XData(i).addRow(xM);

    for j=2:numOfNodes
        XData(i) = XData(i).addRow(xM);
    end

    % Генерация Y
    yM = zeros(1,numOfWhiteNodes);
    for j = 1:numOfWhiteNodes
        temp = XData(i).getRow(j);
         yM(j) = 20 + 10 * temp(1) + 5 * temp(2);
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
trainer = Trainer(modelShell, 10);
%% Вызываем настройщик (не указываем целевые индексы = используем все белые вершины)
trainer.Train(XDataTrain, YDataTrain, XDataTest, YDataTest, 0.1, 0.9, 0.99, 1e-8, NodeSize, NodeWeight, 200, 1e12, -1e12, 1.0, 1.0, 0.5, [], 'mape');
%% Тестирование модели

whiteIdx = modelShell.GetWhiteNodesIndices;

numTestSamples = numel(YDataTest);
actualValue = zeros(1, numTestSamples);
predictionValue = zeros(1, numTestSamples);

for j = 1 : numTestSamples
    modelShell.Forward(XDataTest(j));
    actual = YDataTest(j).getRow(1);
    predict = modelShell.GetModelResults();
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

XDataValid = repmat(BWMatrix(), validSamples, 1);
YDataValid = repmat(BWMatrix(), validSamples, 1);

for i = 1:validSamples
    % Генерация Х
    xM = zeros(numInputParams,1);
    for k = 1:numInputParams
        xM(k) = randi([1,20]);
    end
    XDataValid(i) = XDataValid(i).addRow(xM);

    for j=2:numOfNodes
        XDataValid(i) = XDataValid(i).addRow(xM);
    end

    % Генерация Y
    yM = zeros(1,numOfWhiteNodes);
    for j = 1:numOfWhiteNodes
        temp = XData(i).getRow(j);
         yM(j) =20 + 10 * temp(1) + 5 * temp(2);
    end
    YDataValid(i) = YDataValid(i).addRow(yM);
end


validValue = zeros(1,validSamples);
predictOnValid = zeros(1,validSamples);

for j = 1:validSamples
    modelShell.Forward(XDataValid(j));
    forecast = modelShell.GetModelResults();
    actual = YDataValid(j).getRow(1);
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