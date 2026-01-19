# Экспериментальная библиотека для модели черно-белого графа (v.0.2)
## Содержание
- [Структура проекта](#структура-проекта)
- [Пример использования](#пример-использования)
- [Спецификация основных методов](#спецификация-основных-методов)
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
    │   │       BetaGenerator.m             # Пример класса для инициализации параметров бета
    │   │       IRandomGen.m                # Интерфейс, имплементация которого обязательна для всех будущих генераторов
    │   │
    │   └───+Trainer                        # Пространство имен, содержащие оболочку, контролирующую настройку модели
    │           Trainer.m                   # Класс, контролирующий настройку модели
    │
    ├───+coreFunctions                      # Пространство имен, содержащие пример ядровых функций и интерфейс для них
    │       ICoreF.m                        # Интерфейс ядровых функций, имплементация которого обязательна для всех будущих ядровых функций
    │       ...
    └───    LinearFunction.m                # Пример простой линейной функции ядра
  
```
## Пример использования
### Импортируем пространства имен
```matlab
import BWGraph.*;
import BWGraph.CustomMatrix.*;
import BWGraph.RandomGenerator.*;
import BWGraph.Trainer.*;
import coreFunctions.*;
```
### Создаем функции ядра и вершины графа
```matlab
rng(123);
CoreF = LinearFunction();

alfaGen = AlphaGenerator(0.9);
betaGen = BetaGenerator(2);

nodeA = Node(1, 100, "White", CoreF);
nodeB = Node(2, 100, "Black", []);
nodeC = Node(3, 100, "Black", []);

nodeA.addEdge(nodeC);
nodeA.addEdge(nodeB);
nodeB.addEdge(nodeB);
```
### Задаем индивидуальные параметры для вершин
```matlab
NodeSize = [1.5 0.5 0.5]; % Коэффициенты 
NodeWeight = [1.5 0.5 0.5]; % Весовые коэффициенты вершин
```
### Создаем черно-белый граф
```matlab
modelShell = GraphShell(alfaGen, betaGen, nodeA, nodeB, nodeC);
```
### Выполняем генерацию данных
#### Инициализация BWMatrix
```matlab
% Задаем количество синтетических данных
numSamples = 500;
% Считываем количество вершин в графе
numOfNodes = numel(modelShell.ListOfNodes);
% Получаем количество белых вершин
numOfWhiteNodes = modelShell.GetNumOfWhiteNode;
% Получаем количество входных параметров ядровой функции
numInputParams = CoreF.GetNumOfInputParams();
% Создаем объекты BWMatrix для входных (XData) и выходных (YData) подмножеств
XData = repmat(BWMatrix(), numSamples, 1);
YData = repmat(BWMatrix(), numSamples, 1);
```
#### Генерация синтетических данных
```matlab
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
         yM(j) = 10*XData(i).getRow(j) - 51;
    end
    YData(i) = YData(i).addRow(yM);
end
```
#### Делим исходные данные на обучающее и тестовое подмножества
```matlab
% Создаем индикатор из случайных значений
indices = randperm(numSamples);
% Создаем разделитель
splitPoint = round(0.8 * numSamples);
% Извлекаем индексы для обучения и тестирования
trainIndices = indices(1:splitPoint);
testIndices = indices(splitPoint+1:end);

% Отделяем обучающую и тестовую выборку
XDataTrain = XData(trainIndices);
YDataTrain = YData(trainIndices);
XDataTest = XData(testIndices);
YDataTest = YData(testIndices);
```
### (Опционально) Отображаем полученный граф
```matlab
modelShell.DrawGraph("Структура модели до настройки");
```
### Создаем настройщик для модели
```matlab
trainer = Trainer(modelShell, 30);
```
### Запускаем процесс настройки модели
```matlab
trainer.Train(XDataTrain, YDataTrain, XDataTest, YDataTest, 0.1, 0.9, 0.99, 1e-8, NodeSize, NodeWeight, 1000, 1e7, -1e7, 10.0, 0.1, 0, [], 'mae');
```
### Тестируем модель после настройки
```matlab
% Получаем индексы белых вершин
whiteIdx = modelShell.GetWhiteNodesIndices;
% Создаем хранилища для предсказанных значений и фактических значений
numTestSamples = numel(YDataTest);
actualValue = zeros(1, numTestSamples);
predictionValue = zeros(1, numTestSamples);
% Подаем тестовые данные в модель и снимаем результат
for j = 1 : numTestSamples
    modelShell.Forward(XDataTest(j));
    actual = YDataTest(j).getRow(1);
    predict = modelShell.GetModelResults();
    predictionValue(j) = mean(predict(whiteIdx));
    actualValue(j) = mean(actual);
end
```
### (Опционально) Строим график сравнения
```matlab
% Строим график сравнения
figure(...
    'Name', 'Тестирование', ...
    'Position', [10, 10, 900, 500], ...
    'Color', [0.95, 0.95, 0.95], ...
    'Resize', 'off' ...
);
hold on;

% Рисуем линии фактических и модельных значений
plot(1:numTestSamples, actualValue, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Фактические значения');
plot(1:numTestSamples, predictionValue, 'r--s', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Модельные значения');

% Настраиваем график
xlabel('Номер тестового примера');
ylabel('Значение');
title('Апробация модели на тестовом подмножестве');
legend('show', 'Location', 'best');

set(gca, 'FontSize', 14, 'FontWeight', 'bold');

grid on;
hold off;
```

## Спецификация основных методов
### Метод `Train`
Обучает модель на предоставленных данных, используя адаптивный метод оптимизации
Сигнатура:
```matlab
function Train(XDataTrain, YDataTrain, XDataTest, YDataTest, ...
                LearningRate, Beta1, Beta2, Eps, NodeWeight, NodeSize, Epoches, ...
                ClipUp, ClipDown, TargetError, Lambda, Lambda_Agg, targetNodeIndices, errorMetric)
```
Параметры:
| Параметр | Тип | Описание |
| :--- | :--- | :--- |
| **`XDataTrain`** | `BWMatrix` | Обучающая выборка: входные данные. Должен быть массивом `BWGraph.CustomMatrix.BWMatrix`. |
| **`YDataTrain`** | `BWMatrix` | Обучающая выборка: целевые значения. Должен быть массивом `BWGraph.CustomMatrix.BWMatrix`. Размерность должна соответствовать количеству белых узлов в графе. |
| **`XDataTest`** | `BWMatrix` | Тестовая выборка: входные данные. Должен быть массивом `BWGraph.CustomMatrix.BWMatrix`. |
| **`YDataTest`** | `BWMatrix` | Тестовая выборка: целевые значения. Должен быть массивом `BWGraph.CustomMatrix.BWMatrix`. Количество примеров должно совпадать с `XDataTest`. |
| **`LearningRate`** | `double` | Начальная скорость обучения. Положительное конечное число > 0. Динамически уменьшается в процессе обучения. |
| **`Beta1`** | `double` | Коэффициент экспоненциального затухания для первого момента (адаптация градиента). Положительное конечное число > 0. |
| **`Beta2`** | `double` | Коэффициент экспоненциального затухания для второго момента (адаптация learning rate). Положительное конечное число > 0. |
| **`Eps`** | `double` | Малое число для численной стабильности (предотвращение деления на ноль). Положительное конечное число > 0. |
| **`NodeWeight`** | `double (1,:)` | Вектор весов для узлов графа. |
| **`NodeSize`** | `double (1,:)` | Вектор размеров для узлов графа. |
| **`Epoches`** | `integer` | Максимальное количество эпох обучения. Положительное целое число > 0. |
| **`ClipUp`** | `double` | Верхняя граница для ограничения градиента (gradient clipping). Любое конечное число. |
| **`ClipDown`** | `double` | Нижняя граница для ограничения градиента (gradient clipping). Любое конечное число. |
| **`TargetError`** | `double` | Целевое значение ошибки. Обучение остановится, если ошибка станет меньше этого значения. Положительное конечное число > 0. |
| **`Lambda`** | `double` | Коэффициент регуляризации. Строго положительное число > 0. |
| **`Lambda_Agg`** | `double` | Коэффициент регуляризации для агрегирующих функций. Строго положительное число или ноль. |
| **`targetNodeIndices`** | `double[]` | Индексы белых вершин, для которых рассчитывается ошибка. Если не указан, используются все белые вершины. **По умолчанию: `[]`**. Должны быть действительными индексами белых вершин графа. |
| **`errorMetric`** | `string` | Метрика для расчета ошибки. **По умолчанию: `'mae'`**. Допустимые значения: `'mae'` (Mean Absolute Error), `'mape'` (Mean Absolute Percentage Error). |

#### Возвращаемое значение

Метод не возвращает явного значения. Результатом обучения являются обновленные веса модели, которые сохраняются внутри её структуры.

#### Пример использования

```matlab
% Пример вызова метода Train
model.Train(X_train, Y_train, X_test, Y_test, ...
            0.001, 0.9, 0.999, 1e-8, nodeWeights, nodeSizes, 1000, ...
            1.0, -1.0, 0.001, 0.01, 0.005, [], 'mae');
```

### Приватный метод `Compute_V5`
Вызывается в теле метода `Train` и выполняет коррекцию весов пакеным градиентным спуском с использованием ADAM-оптимизатора
Сигнатура:
```matlab
function Compute_V5(obj, XData, YData, Beta1, Beta2, Eps, NodesWeights, NodesLRScale, ClipUp, ClipDown, Lambda, lambda_Agg, targetWhiteIndices,errorMetric)
```
### Параметры

| Параметр | Тип/Ограничения | Описание |
| :--- | :--- | :--- |
| **`obj`** | `BWGraph.Trainer.Trainer` | Экземпляр объекта Trainer, для которого вызывается метод. |
| **`XData`** | - | Входные данные для вычислений. |
| **`YData`** | - | Целевые значения (метки) для вычислений. |
| **`Beta1`** | `double`<br/>`mustBePositive, mustBeFinite` | Коэффициент экспоненциального затухания для оценки первого момента (среднего) градиентов. |
| **`Beta2`** | `double`<br/>`mustBePositive, mustBeFinite` | Коэффициент экспоненциального затухания для оценки второго момента (нецентрированной дисперсии) градиентов. |
| **`Eps`** | `double`<br/>`mustBePositive, mustBeFinite` | Малое число для численной стабильности, предотвращающее деление на ноль. |
| **`NodesWeights`** | `double (1,:)` | Вектор весов вершин. Используется для масштабирования степени настройки параметров отдельных вершин. |
| **`NodesLRScale`** | `double (1,:)` | Вектор коэффициентов масштабирования скорости обучения (Learning Rate) для отдельных вершин. |
| **`ClipUp`** | `double`<br/>`mustBeFinite` | Верхняя граница для ограничения (clipping) значений градиентов. |
| **`ClipDown`** | `double`<br/>`mustBeFinite` | Нижняя граница для ограничения (clipping) значений градиентов. |
| **`Lambda`** | `double`<br/>`mustBePositive` | Коэффициент регуляризации (L2-регуляризация или подобная). |
| **`lambda_Agg`** | `double`<br/>`mustBeNonnegative` | Коэффициент влияния усредненной ошибки. Регуляризационный параметр для агрегирующих функций. |
| **`targetWhiteIndices`** | `double (1,:)` | Вектор индексов целевых белых вершин, для которых рассчитывается ошибка и производные. |
| **`errorMetric`**| `string`| Разновидность целевой функции для расчета ошибок в белых вершинах. Доступные целевые функции: 'mae', 'mse', 'rmse' |

### Особенности реализации

*   Метод использует **аргументные блоки (arguments block)** MATLAB для валидации входных параметров.
*   Реализует механизм **ограничения градиентов (gradient clipping)** через параметры `ClipUp` и `ClipDown`.
*   Поддерживает **адаптивную скорость обучения** для отдельных вершин через параметр `NodesLRScale`.
*   Включает **два типа регуляризации**: основную (`Lambda`) и для агрегирующих функций (`lambda_Agg`).






















