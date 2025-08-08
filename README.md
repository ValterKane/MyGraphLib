# Экспериментальная библиотека для модели черно-белого графа (v.0.2)
## Структура проекта:
```
    MyGraphLib
    
    ├───+BWGraph                            # Пространство имен модели черно-белого графа
    │   │   Edge.m                          # Класс, описывающие рёбра графа
    │   │   GraphShell.m                    # Класс-оболочка графа
    │   │   Node.m                          # Класс, описывающий вершины графа
    │   │   NodeColor.m                     # Перечисление для определения типа вершины (черная / белая)
    │   │
    │   ├───+CustomMatrix                   # Пространство имен для описания матрицы из векторов произвольной длины
    │   │       BWMatrix.m                  # Класс, описывающий матрицу и базовые операции
    │   │       BWRow.m                     # Класс, описывающий вектор-строку произвольной длины
    │   │
    │   ├───+RandomGenerator                # Пространство имен для генераторов первичной инициализации параметров рёбер
    │   │       AlphaGenerator.m            # Пример класса для инициализации параметров альфа
    │   │       BetaGenerator.m             # пример класса для инициализации параметров бета
    │   │       IRandomGen.m                # Интерфейс, имплементация которого обязательна для всех будущих генераторов
    │   │
    │   └───+Trainer                        # Пространство имен, содержащие оболочку, контролирующую настройку модели
    │           Trainer.m                   # Класс, контролирующий настройку модели
    │
    ├───+coreFunctions                      # Пространство имен, содержащие пример ядровых функций и интерфейс для них
    │       HeatTransferBC.m                # Пример ядровой функции на основе уравнения теплового баланса
    │       ICoreF.m                        # Интерфейс ядровых функций, имплементация которого обязательна для всех будущих ядровых функций
    │       LinearFunction.m                # Пример простой линейной функции ядра
    │       LinearRegression.m              # Пример регрессионной функции ядра
    │       SigmoidFunction.m               # Пример сигмоидальной функции ядра
    └───    SimpleAddingCoreFunction.m      # Пример суммирующей функции ядра
  
```
## Пример использования
```matlab
%% Генерация данных
import BWGraph.*;
import BWGraph.CustomMatrix.*;
import BWGraph.RandomGenerator.*;
import BWGraph.Trainer.*;
import coreFunctions.*;

rng(2);
% Общая функция для всех вершин
CoreF = LinearFunction();
alfaGen = AlphaGenerator(0.9);
betaGen = BetaGenerator(1);

nodeA = Node(1, 30, "White", CoreF);
nodeB = Node(2, 30, "Black", []);
nodeC = Node(3, 30, "Black", []);
nodeD = Node(4, 30, "Black", []);

nodeA.addEdge(nodeB);
nodeA.addEdge(nodeC);
nodeA.addEdge(nodeD);

% Индивидуальные параметры для вершин (общие для всех экспериментов)
NodeSize = [1 1 1 1]; % Коэффициенты 
NodeWeight = [1 1 1 1]; % Весовые коэффициенты вершин

% Создаем графовую модель
modelShell = GraphShell(alfaGen, betaGen,  nodeA, nodeB, nodeC, nodeD);

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
    x = randi([-10,10]);
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
         yM(j) = 10*XData(i).getRow(j) - 50;
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

%% Строим диаграммы распределения (опционально)
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

legend(labels, 'Location', 'best', 'FontSize', 12);

%% Отрисовка структуры графа
modelShell.DrawGraph("Структура модели до настройки");

%% Настройка модели
% Задаем настройщик
trainer = Trainer(modelShell, 30);
% Вызываем настройщик
trainer.Train(XDataTrain, YDataTrain, XDataTest, YDataTest, 0.1, 0.1, 0.1, 1e-8, NodeSize, NodeWeight, 200, 1e12, -1e12, 1.0, 1.0);

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
    xM = zeros(numInputParams,1);
    for j = 1:numOfNodes
        for k = 1:numInputParams
            xM(k) = randi([20,30]);
        end
        XDataValid(i) = XDataValid(i).addRow(xM);
    end 
end

for i = 1:validSamples
    yM = zeros(1,numOfWhiteNodes);
    for j = 1:numOfWhiteNodes
         yM(j) = 10*XDataValid(i).getRow(j)-50;
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
fprintf('Результирующее MAE на валидации %2.3f\n', mae(predictOnValid,validValue))
fprintf('Результирующее MAPE на валидации %2.3f\n', mape(validValue, predictOnValid))

fprintf('Результирующее MAE на тесте %2.3f\n', mae(predictionValue,actualValue))
fprintf('Результирующее MAPE на тесте %2.3f\n', mape(actualValue,predictionValue))

%% Отобразить граф
modelShell.DrawGraph("Структура модели после настройки");
```
