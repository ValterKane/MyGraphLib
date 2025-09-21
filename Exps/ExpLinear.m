%% Генерация данных
import BWGraph.*;
import BWGraph.CustomMatrix.*;
import BWGraph.RandomGenerator.*;
import BWGraph.Trainer.*;
import coreFunctions.*;
% 24 для модели 4-х вершин
% 42 для модели 10х-51=y и 5x+20=y
% 5 для модели 5-и вершин на 10х-51=y
rng(123);
% Общая функция для всех вершин
CoreF = LinearFunction();
alfaGen = AlphaGenerator(0.9);
betaGen = BetaGenerator(2);

nodeA = Node(1, 100, "White", CoreF);
nodeB = Node(2, 100, "Black", []);
nodeC = Node(3, 100, "Black", []);

nodeA.addEdge(nodeC);
nodeA.addEdge(nodeB);
nodeB.addEdge(nodeB);

% Индивидуальные параметры для вершин (общие для всех экспериментов)
NodeSize = [1.5 0.5 0.5 0.5 0.5]; % Коэффициенты 
NodeWeight = [1.5 0.5 0.5 0.5 0.5]; % Весовые коэффициенты вершин

% Создаем графовую модель
modelShell = GraphShell(alfaGen, betaGen, nodeA, nodeB, nodeC);

%% Создаем входные данные
% Генерация данных с учетом индивидуальных характеристик вершин
numSamples = 500;
numOfNodes = numel(modelShell.ListOfNodes);
numOfWhiteNodes = modelShell.GetNumOfWhiteNode; % Получаем количество вершин
numInputParams = CoreF.GetNumOfInputParams(); % Получаем количество входных параметров
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
         yM(j) = 10*XData(i).getRow(j) - 51;
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

%% Вызываем настройщик
trainer.Train(XDataTrain, YDataTrain, XDataTest, YDataTest, 0.1, 0.9, 0.99, 1e-8, NodeSize, NodeWeight, 1000, 1e7, -1e7, 10.0, 0.1, 0, [], 'mae');

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
%% Генерируем данные для валидации
validSamples = 50;
numOfNodes = numel(modelShell.ListOfNodes);
numOfWhiteNodes = modelShell.GetNumOfWhiteNode; % Получаем количество вершин
numInputParams = CoreF.GetNumOfInputParams(); % Получаем количество входных параметров

rng(123);
XDataValid = repmat(BWMatrix(), validSamples, 1);
YDataValid = repmat(BWMatrix(), validSamples, 1);
validX = zeros(1,validSamples);

for i = 1:validSamples
    validX(i) = randi([1,100]);
end

for i = 1:validSamples 
    xM = zeros(numInputParams,1);
    for j = 1:numOfNodes
        for k = 1:numInputParams
            xM(k) = validX(i);
        end
        XDataValid(i) = XDataValid(i).addRow(xM);
    end 
end

validValue = zeros(1,validSamples);
for i = 1:validSamples
    yM = zeros(1,numOfWhiteNodes);
    for j = 1:numOfWhiteNodes
         yM(j) = 10*XDataValid(i).getRow(j)-51;
    end
    YDataValid(i) = YDataValid(i).addRow(yM);
    validValue(i) = mean(yM);
end
%% Валидация
predictOnValid = zeros(1,validSamples);

for j = 1:validSamples
    modelShell.Forward(XDataValid(j));
    forecast = modelShell.GetModelResults();
    predictOnValid(j) = mean(forecast(whiteIdx));
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
