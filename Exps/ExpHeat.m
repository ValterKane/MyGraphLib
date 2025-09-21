%% Подготовка
import BWGraph.*;
import BWGraph.CustomMatrix.*;
import BWGraph.RandomGenerator.*;
import BWGraph.Trainer.*;

HeatBC = coreFunctions.PlateHeatingModel(320.0, 100, 50, 1e-5, 2, 293, 1000);
alfaGen = AlphaGenerator(0.9);
betaGen = BetaGenerator(1);

%% Создание и настройка модели
nodeA = Node(1, 200,'White',HeatBC);
nodeB = Node(2, 200,'Black',HeatBC);
nodeC = Node(3, 200,'Black',HeatBC);

nodeB.addEdge(nodeA);
nodeC.addEdge(nodeA);

nodeA.addEdge(nodeB);
nodeA.addEdge(nodeC);

% Создаем графовую модель
modelShell = GraphShell(alfaGen,betaGen,nodeA,nodeB,nodeC);

% Индивидуальные параметры для вершин (общие для всех экспериментов)
NodeSize = [1 1 1]; % Коэффициенты 
NodeWeight = [1 1 1]; % Весовые коэффициенты вершин

%% Генерация данных и подготовка подвыборок
% Генерация данных с учетом индивидуальных характеристик вершин
numSamples = 500;
numOfNodes = numel(modelShell.ListOfNodes);
numOfWhiteNodes = modelShell.GetNumOfWhiteNode; % Получаем количество вершин
numInputParams = HeatBC.GetNumOfInputParams(); % Получаем количество входных параметров
rng(22);
% Генерация данных
timeRange_for_v1 = [1000, 11000];
TinfRange_for_v1 = [900, 1200];

timeRange_for_v2 = [1000, 15000];
TinfRange_for_v2 = [800, 1150];  

timeRange_for_v3 = [1000, 9000];
TinfRange_for_v3 = [900, 1100];  

% Случайные значения времени и температуры окружающей среды (значения X)
timeValues_for_v1 = rand(numSamples, 1) * (timeRange_for_v1(2) - timeRange_for_v1(1)) + timeRange_for_v1(1);
TinfValues_for_v1 = rand(numSamples, 1) * (TinfRange_for_v1(2) - TinfRange_for_v1(1)) + TinfRange_for_v1(1);

timeValues_for_v2 = rand(numSamples, 1) * (timeRange_for_v2(2) - timeRange_for_v2(1)) + timeRange_for_v2(1);
TinfValues_for_v2 = rand(numSamples, 1) * (TinfRange_for_v2(2) - TinfRange_for_v2(1)) + TinfRange_for_v2(1);

timeValues_for_v3 = rand(numSamples, 1) * (timeRange_for_v3(2) - timeRange_for_v3(1)) + timeRange_for_v3(1);
TinfValues_for_v3 = rand(numSamples, 1) * (TinfRange_for_v3(2) - TinfRange_for_v3(1)) + TinfRange_for_v3(1);


% Зададим синтетические значения средней температуры, смещенной
% относительно нагрева соседних вершин
TavgValues = zeros(numSamples, 1);
for i = 1:numSamples
    inputParams_for_v1 = [timeValues_for_v1(i); TinfValues_for_v1(i)];
    inputParams_for_v2 = [timeValues_for_v2(i); TinfValues_for_v2(i)];
    inputParams_for_v3 = [timeValues_for_v3(i); TinfValues_for_v3(i)];
    
    
    TavgValues(i) = HeatBC.CalcCoreFunction(inputParams_for_v1) + ...
        0.1 * HeatBC.CalcCoreFunction(inputParams_for_v2) + ...
        0.15 *HeatBC.CalcCoreFunction(inputParams_for_v3);
end

% Определим матрицы входа и выхода
XData = repmat(BWMatrix(), numSamples, 1);
YData = repmat(BWMatrix(), numSamples, 1);

for i = 1:numSamples
    inputParams_for_v1 = [timeValues_for_v1(i); TinfValues_for_v1(i)];
    inputParams_for_v2 = [timeValues_for_v2(i); TinfValues_for_v2(i)];
    inputParams_for_v3 = [timeValues_for_v3(i); TinfValues_for_v3(i)];
    XData(i) = XData(i).addRow(inputParams_for_v1);
    XData(i) = XData(i).addRow(inputParams_for_v2);
    XData(i) = XData(i).addRow(inputParams_for_v3);
end

for i = 1:numSamples
    yMatrix = zeros(1,numOfWhiteNodes);
    for j = 1:numOfWhiteNodes
        yMatrix(j) = TavgValues(i);
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

%% Создаем настройщик
trainer = Trainer(modelShell, 30);
%% Настройка
trainer.Train(XDataTrain, YDataTrain, XDataTest, YDataTest, 0.1, 0.9, 0.99,1e-8, NodeSize, NodeWeight, 100, 1e7, -1e7, 6.0, 0.4, 0.1, [], 'mae');
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

%% Валидация
validSamples = 50;

% Генерация данных
timeRange_for_v1 = [1000, 11000];
TinfRange_for_v1 = [900, 1200];

timeRange_for_v2 = [1000, 15000];
TinfRange_for_v2 = [800, 1150];  

timeRange_for_v3 = [1000, 9000];
TinfRange_for_v3 = [900, 1100];  

% Случайные значения времени и температуры окружающей среды (значения X)
timeValues_for_v1 = rand(validSamples, 1) * (timeRange_for_v1(2) - timeRange_for_v1(1)) + timeRange_for_v1(1);
TinfValues_for_v1 = rand(validSamples, 1) * (TinfRange_for_v1(2) - TinfRange_for_v1(1)) + TinfRange_for_v1(1);

timeValues_for_v2 = rand(validSamples, 1) * (timeRange_for_v2(2) - timeRange_for_v2(1)) + timeRange_for_v2(1);
TinfValues_for_v2 = rand(validSamples, 1) * (TinfRange_for_v2(2) - TinfRange_for_v2(1)) + TinfRange_for_v2(1);

timeValues_for_v3 = rand(validSamples, 1) * (timeRange_for_v3(2) - timeRange_for_v3(1)) + timeRange_for_v3(1);
TinfValues_for_v3 = rand(validSamples, 1) * (TinfRange_for_v3(2) - TinfRange_for_v3(1)) + TinfRange_for_v3(1);


% Зададим синтетические значения средней температуры, смещенной
% относительно нагрева соседних вершин
TavgValues = zeros(numSamples, 1);
for i = 1:validSamples
    inputParams_for_v1 = [timeValues_for_v1(i); TinfValues_for_v1(i)];
    inputParams_for_v2 = [timeValues_for_v2(i); TinfValues_for_v2(i)];
    inputParams_for_v3 = [timeValues_for_v3(i); TinfValues_for_v3(i)];
    
    TavgValues(i) = HeatBC.CalcCoreFunction(inputParams_for_v1) + ...
        0.1 * HeatBC.CalcCoreFunction(inputParams_for_v2) + ...
        0.15 *HeatBC.CalcCoreFunction(inputParams_for_v3);
end

% Определим матрицы входа и выхода
XDataValid = repmat(BWMatrix(), validSamples, 1);
YDataValid = repmat(BWMatrix(), validSamples, 1);

for i = 1:validSamples
    inputParams_for_v1 = [timeValues_for_v1(i); TinfValues_for_v1(i)];
    inputParams_for_v2 = [timeValues_for_v2(i); TinfValues_for_v2(i)];
    inputParams_for_v3 = [timeValues_for_v3(i); TinfValues_for_v3(i)];
    XDataValid(i) = XDataValid(i).addRow(inputParams_for_v1);
    XDataValid(i) = XDataValid(i).addRow(inputParams_for_v2);
    XDataValid(i) = XDataValid(i).addRow(inputParams_for_v3);
end

for i = 1:validSamples
    yMatrix = zeros(1,numOfWhiteNodes);
    for j = 1:numOfWhiteNodes
        yMatrix(j) = TavgValues(i);
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
figure(...
    'Name', 'Валидация', ...
    'Position', [10, 10, 900, 500], ...
    'Color', [0.95, 0.95, 0.95], ...
    'Resize', 'off' ...
);
hold on;

% Рисуем линии фактических и модельных значений
plot(1:validSamples, validValue, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Фактические значения');
plot(1:validSamples, predictOnValid, 'r-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Модельные значения');

% Настраиваем график
xlabel('Номер примера');
ylabel('Значение');
title('Апробация модели на валидационном подмножестве');
legend('show', 'Location', 'best');

set(gca, 'FontSize', 14, 'FontWeight', 'bold');

grid on;
hold off;
%% Анализ нормальности остатков
residual = validValue - predictOnValid;

% Тест на нормальность (Lilliefors)
[h, p] = lillietest(residual);
if h == 0
    fprintf('Остатки нормально распределены (p=%.4f)\n', p);
else
    fprintf('Остатки НЕ нормальны (p=%.4f)\n', p);
end

%% Тест Голдфельда-Квандта на гомоскедастичность остатков
residual = validValue - predictOnValid;

% 1. Сортируем данные по предсказанным значениям (или по одной из переменных)
[sorted_y_pred, sort_idx] = sort(predictOnValid);
sorted_residuals = residual(sort_idx);

% 2. Разделяем остатки на 3 группы (исключая среднюю часть)
n = length(residual);
k = floor(n / 3); % Размер групп

% Первая группа (наименьшие ŷ)
residuals_low = sorted_residuals(1:k);

% Последняя группа (наибольшие ŷ)
residuals_high = sorted_residuals(end-k+1:end);

% 3. Сравниваем дисперсии (F-тест)
var_low = var(residuals_low);
var_high = var(residuals_high);

% F-статистика
F_stat = var_high / var_low; % Берем большую дисперсию в числитель

% Критическое значение F-распределения (для alpha=0.05)
df = k - 1;
F_critical = finv(0.95, df, df);

fprintf('F_stat = %.4f\n', F_stat);
fprintf('F_crit = %.4f\n', F_critical);
fprintf('P_value = %.4f\n', F_stat / F_critical);

% Проверка гипотезы
if F_stat > F_critical
    disp('Гетероскедастичность (p < 0.05)');
else
    disp('Гомоскедастичность (p > 0.05)');
end
%% Анализ метрики R2

SS_residual = sum((validValue - predictOnValid).^2);       
SS_total = sum((validValue - mean(actualValue)).^2); 
R2 = 1 - (SS_residual / SS_total);  

fprintf('R² на валидации = %.4f\n', R2);
    
SS_residual = sum((actualValue - predictionValue).^2);       
SS_total = sum((actualValue - mean(actualValue)).^2); 
R2 = 1 - (SS_residual / SS_total);

fprintf('R² на тесте = %.4f\n', R2);

%% Анализ метрики MAE
fprintf('Результирующее MAE на валидации %2.3f\n', mae(predictOnValid,validValue))
fprintf('Результирующее MAE на тесте %2.3f\n', mae(predictionValue,actualValue))

%% Анализ дисперсий
fprintf('D[Y_val]: %3f\n', var(predictOnValid))
fprintf('D[Y_val_actual] %3f\n', var(validValue))
fprintf('D[Y_test]: %3f\n', var(predictionValue))
fprintf('D[Y_test_actual]: %3f\n', var(actualValue))

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