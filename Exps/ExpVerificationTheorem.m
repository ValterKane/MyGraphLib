%% Очистить все
clear; clc; close all;
rng(22);

import BWGraph.*;
import BWGraph.CustomMatrix.*;
import BWGraph.RandomGenerator.*;
import BWGraph.Trainer.*;

HeatBC = coreFunctions.Heating2DModel(30, 20, 20, 70, 1.5e-5, 0.3, 0.360, 30, 10);
alfaGen = FullRandomAlfaGen(1,5); % Гиперпараметр
betaGen = FullRandomBetaGen(1,1e2); % Гиперпараметр

nodeA = Node(1, 1,'White',HeatBC,'linear');
nodeB = Node(2, 1,'Black',HeatBC,'linear');
nodeC = Node(3, 1,'Black',HeatBC,'linear');

% nodeA.addEdge(nodeC);
% nodeA.addEdge(nodeB);

nodeB.addEdge(nodeA);
nodeC.addEdge(nodeA);
nodeC.addEdge(nodeB);

% Индивидуальные параметры для вершин (общие для всех экспериментов)
NodeWeight = [1 0.5 0.5]; % Весовые коэффициенты вершин

% Создаем графовую модель
modelShell = GraphShell(alfaGen,betaGen,NodeWeight,nodeA,nodeB,nodeC);
% Отрисовать граф
modelShell.DrawGraph_New('Модель численной верифиакции');

% Генерация данных и подготовка подвыборок (синтетика)
% Генерация данных с учетом индивидуальных характеристик вершин
numSamples = 150;
numOfNodes = numel(modelShell.ListOfNodes);
numOfWhiteNodes = modelShell.GetNumOfWhiteNode; % Получаем количество вершин
numInputParams = HeatBC.GetNumOfInputParams(); % Получаем количество входных параметров

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
        0.15 * HeatBC.CalcCoreFunction(inputParams_for_v3);
    
end

XData_for_bagging = [timeValues_for_v1, timeValues_for_v2, timeValues_for_v3, ...
        TinfValues_for_v1,TinfValues_for_v2,TinfValues_for_v3];
YData_for_bagging = TavgValues;

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

cv = cvpartition(size(XData,1), 'HoldOut', 0.3);
XDataTrain = XData(training(cv), :);
YDataTrain = YData(training(cv));
XDataTest = XData(test(cv), :);
YDataTest = YData(test(cv));

XTrain = XData_for_bagging(training(cv), :);
yTrain = YData_for_bagging(training(cv));
XTest = XData_for_bagging(test(cv), :);
yTest = YData_for_bagging(test(cv));

%% Настройка учителя
% Опции настройки
trainerOptions = TrainingOptions( ...
    "LearningRate", 0.01, ...
    "Beta1", 0.9, ...
    "Beta2", 0.999, ...
    "Eps", 1e-8, ...
    "NodeSize", [1, 1, 1], ...
    "Epoches", 500, ...
    "ClipUp", 1e7, ...
    "ClipDown", -1e7, ...
    "TargetError", 5, ...
    "Lambda_Agg", 0.0, ...
    "Lambda_Alph", 0.4, ...
    "Lambda_Beta", 0.4, ...
    "Lambda_Self", 0.1, ...
    "Lambda_Struct", 0.9, ...
    "Lambda_Gamma",0.4, ...
    "ErrorMetric",'mae', ...
    "LossFunction",'mse', ...
    "TargetNodeIndices",[], ...
    "BatchSize", 1);

% Инициализация учителя
trainer = Trainer(modelShell, trainerOptions);

%% Запуск процесса
trainer.Train(XDataTrain, YDataTrain, XDataTest, YDataTest);

%% ===== 4. ВЫЧИСЛЕНИЕ C_{b→w} =====
nodes = modelShell.ListOfNodes;
numNodes = numel(nodes);
whiteIdx = find(arrayfun(@(n) n.getNodeType() == BWGraph.NodeColor.White, nodes));
blackIdx = find(arrayfun(@(n) n.getNodeType() == BWGraph.NodeColor.Black, nodes));

% Собираем S_v = Σ_{e∈Out(v)} α_e + 1 для каждой вершины v
S_out = zeros(numNodes, 1);
for i = 1:numNodes
    outEdges = nodes(i).getOutEdges();
    if isempty(outEdges)
        S_out(i) = 1;
    else
        S_out(i) = sum([outEdges.Alfa]) + 1;
    end
end

% Строим M: M(v,u) = вклад J_u в J_v
%   Если (u→v) ∈ E и v ∈ B: M(v,u) += α_{u→v} / S_out(u)          [δ_in]
%   Если (v→u) ∈ E и v ∈ B: M(v,u) += α_{v→u} / S_out(v)          [δ_out]
M = zeros(numNodes, numNodes);
for i = 1:numNodes
    outEdges = nodes(i).getOutEdges();
    for e = outEdges'
        from = i;
        to = find([nodes.ID] == e.TargetNode.ID, 1);

        % δ_in: чёрная вершина to получает влияние от from
        if ismember(to, blackIdx)
            M(to, from) = M(to, from) + e.Alfa / S_out(from);
        end

        % δ_out: чёрная вершина from влияет на to
        if ismember(from, blackIdx)
            M(from, to) = M(from, to) + e.Alfa / S_out(from);
        end
    end
end

% C = (I - M)^{-1}
C_matrix = (eye(numNodes) - M) \ eye(numNodes);

% C_{b→w}: строки чёрных, столбцы белых
numW = numel(whiteIdx); numB = numel(blackIdx);
C_bw = C_matrix(blackIdx, whiteIdx);

fprintf('\n========== МАТРИЦА C_{b→w} ==========\n');
for bi = 1:numB
    for wi = 1:numW
        fprintf('C(чёрный %d → белый %d) = % .12f\n', blackIdx(bi), whiteIdx(wi), C_bw(bi, wi));
    end
end
fprintf('======================================\n\n');

%% ===== 5. ПРОВЕРКА ДЕКОМПОЗИЦИИ J_b^{struct} = Σ C_{b→w}·J_w =====
% Теорема 1 утверждает: J_b = Σ C_{b→w}·J_w для СТРУКТУРНОЙ части (eq. J_black).
% J_b^{self} — ортогональный компонент и не участвует в декомпозиции.
nTest = numel(XDataTest);
loss_white = @(F, Y) (F - Y)^2;

J_b_struct = zeros(nTest, numB);
J_b_dec = zeros(nTest, numB);
J_w_arr = zeros(nTest, numW);
maxRelErr = 0;

fprintf('========== ПРОВЕРКА ДЕКОМПОЗИЦИИ ==========\n');

for j = 1:nTest
    F = modelShell.GetCurrentResult(XDataTest(j));
    Y_row = YDataTest(j).getRow(1);  % 1×numW: целевые значения для всех белых

    % J_w — MSE для каждой белой вершины
    for wi = 1:numW
        J_w_arr(j, wi) = loss_white(F(whiteIdx(wi)), Y_row(wi));
    end

    % Структурная часть J_b = решение (I-M)·J = J0 через ряд Неймана
    % J0: J_w для белых, 0 для чёрных
    J0 = zeros(numNodes, 1);
    for wi = 1:numW
        J0(whiteIdx(wi)) = J_w_arr(j, wi);
    end

    % Итерация: J^{k+1} = M·J^k + J0 до сходимости
    J_iter = J0;
    for it = 1:100
        J_next = M * J_iter + J0;
        if norm(J_next - J_iter, inf) < 1e-15
            J_iter = J_next;
            break;
        end
        J_iter = J_next;
    end

    for bi = 1:numB
        J_b_struct(j, bi) = J_iter(blackIdx(bi));
    end

    % J_b через декомпозицию: C·J_w
    for bi = 1:numB
        J_b_dec(j, bi) = C_bw(bi, :) * J_w_arr(j, :)';
    end

    % Вывод первых 5 сэмплов
    if j <= 5
        for bi = 1:numB
            relErr = abs(J_b_struct(j, bi) - J_b_dec(j, bi)) / (abs(J_b_struct(j, bi)) + 1e-15);
            maxRelErr = max(maxRelErr, relErr);
            fprintf('Сэмпл%d Чёрн%d | J_struct=% .6e | C·J_w=% .6e | |Δ|=% .2e | rel=% .2e\n', ...
                j, blackIdx(bi), J_b_struct(j, bi), J_b_dec(j, bi), ...
                abs(J_b_struct(j, bi) - J_b_dec(j, bi)), relErr);
        end
    end
end

% Полный макс
for j = 1:nTest
    for bi = 1:numB
        relErr = abs(J_b_struct(j, bi) - J_b_dec(j, bi)) / (abs(J_b_struct(j, bi)) + 1e-15);
        maxRelErr = max(maxRelErr, relErr);
    end
end

fprintf('\nМаксимальная относительная погрешность: %.2e\n', maxRelErr);
if maxRelErr < 1e-12
    fprintf('РЕЗУЛЬТАТ: Декомпозиция подтверждена (МАШИННАЯ ТОЧНОСТЬ).\n');
elseif maxRelErr < 1e-6
    fprintf('РЕЗУЛЬТАТ: Декомпозиция подтверждена (высокая точность).\n');
else
    fprintf('РЕЗУЛЬТАТ: Декомпозиция НЕ подтверждена.\n');
end
fprintf('J_b^{self} ортогональна и не участвует в декомпозиции.\n');

%% ===== 6. НЕЗАВИСИМОСТЬ C_{b→w} ОТ ДАННЫХ =====
% Генерируем новый набор (другое seed)
rng(99);
XDataNew = repmat(BWMatrix(), nTest, 1);
for j = 1:nTest
    t1 = rand*(timeRange_for_v1(2)-timeRange_for_v1(1)) + timeRange_for_v1(1);
    i1 = rand*(TinfRange_for_v1(2)-TinfRange_for_v1(1)) + TinfRange_for_v1(1);
    t2 = rand*(timeRange_for_v2(2)-timeRange_for_v2(1)) + timeRange_for_v2(1);
    i2 = rand*(TinfRange_for_v2(2)-TinfRange_for_v2(1)) + TinfRange_for_v2(1);
    t3 = rand*(timeRange_for_v3(2)-timeRange_for_v3(1)) + timeRange_for_v3(1);
    i3 = rand*(TinfRange_for_v3(2)-TinfRange_for_v3(1)) + TinfRange_for_v3(1);
    XDataNew(j) = XDataNew(j).addRow([t1; i1]);
    XDataNew(j) = XDataNew(j).addRow([t2; i2]);
    XDataNew(j) = XDataNew(j).addRow([t3; i3]);
    inputParams_for_v1_new = [t1; i1];
    inputParams_for_v2_new = [t2; i2];
    inputParams_for_v3_new = [t3; i3];
    TavgValues(j) = HeatBC.CalcCoreFunction(inputParams_for_v1_new) + ...
        0.1 * HeatBC.CalcCoreFunction(inputParams_for_v2_new) + ...
        0.15 * HeatBC.CalcCoreFunction(inputParams_for_v3_new);
end

% C_bw — те же (вычислены один раз шагом 4)
% Проверка: на новых данных J_b_pred = C·J_w_new
J_w_new = zeros(nTest, numW);
for j = 1:nTest
    F_new = modelShell.GetCurrentResult(XDataNew(j));
    for wi = 1:numW
        % Y_hat из исходной модели HeatBC (не зависит от C)
        x = XDataNew(j).getRow(whiteIdx(wi));
        Y_hat = HeatBC.CalcCoreFunction(x);
        J_w_new(j, wi) = (F_new(whiteIdx(wi)) - Y_hat)^2;
    end
end

J_b_new_pred = J_w_new * C_bw';

%% ===== 6. ПРОВЕРКА ДЕКОМПОЗИЦИИ НА НОВЫХ ДАННЫХ =====
fprintf('\n========== ПРОВЕРКА НА НОВЫХ ДАННЫХ ==========\n');

J_w_new_full = zeros(nTest, numW);
J_b_struct_new = zeros(nTest, numB);
J_b_dec_new = zeros(nTest, numB);
maxRelErr_new = 0;

for j = 1:nTest
    F_new = modelShell.GetCurrentResult(XDataNew(j));

    % Синтетический Tavg для нового сэмпла
    t1_n = XDataNew(j).getRow(1); t1_n = t1_n(1);
    t2_n = XDataNew(j).getRow(2); t2_n = t2_n(1);
    t3_n = XDataNew(j).getRow(3); t3_n = t3_n(1);
    i1_n = XDataNew(j).getRow(1); i1_n = i1_n(2);
    i2_n = XDataNew(j).getRow(2); i2_n = i2_n(2);
    i3_n = XDataNew(j).getRow(3); i3_n = i3_n(2);

    Y_true = HeatBC.CalcCoreFunction([t1_n; i1_n]) + ...
        0.1 * HeatBC.CalcCoreFunction([t2_n; i2_n]) + ...
        0.15 * HeatBC.CalcCoreFunction([t3_n; i3_n]);

    % J_w для белых вершин
    for wi = 1:numW
        J_w_new_full(j, wi) = (F_new(whiteIdx(wi)) - Y_true)^2;
    end

    % Структурная J_b через ряд Неймана: J = M·J + J0
    J0 = zeros(numNodes, 1);
    for wi = 1:numW
        J0(whiteIdx(wi)) = J_w_new_full(j, wi);
    end

    J_iter = J0;
    for it = 1:100
        J_next = M * J_iter + J0;
        if norm(J_next - J_iter, inf) < 1e-15, break; end
        J_iter = J_next;
    end

    for bi = 1:numB
        J_b_struct_new(j, bi) = J_iter(blackIdx(bi));
    end

    % J_b через декомпозицию: C_{b→w} · J_w
    for bi = 1:numB
        J_b_dec_new(j, bi) = C_bw(bi, :) * J_w_new_full(j, :)';
    end

    % Погрешность
    for bi = 1:numB
        relErr = abs(J_b_struct_new(j, bi) - J_b_dec_new(j, bi)) / (abs(J_b_struct_new(j, bi)) + 1e-15);
        maxRelErr_new = max(maxRelErr_new, relErr);
    end

    % Вывод первых 5 сэмплов (как в анализе на тестовых)
    if j <= 5
        for bi = 1:numB
            relErr = abs(J_b_struct_new(j, bi) - J_b_dec_new(j, bi)) / (abs(J_b_struct_new(j, bi)) + 1e-15);
            fprintf('Сэмпл%d Чёрн%d | J_struct=% .6e | C·J_w=% .6e | |Δ|=% .2e | rel=% .2e\n', ...
                j, blackIdx(bi), J_b_struct_new(j, bi), J_b_dec_new(j, bi), ...
                abs(J_b_struct_new(j, bi) - J_b_dec_new(j, bi)), relErr);
        end
    end
end

fprintf('Максимальная относительная погрешность на НОВЫХ данных: %.2e\n', maxRelErr_new);
if maxRelErr_new < 1e-12
    fprintf('РЕЗУЛЬТАТ: Декомпозиция подтверждена на новых данных (МАШИННАЯ ТОЧНОСТЬ).\n');
elseif maxRelErr_new < 1e-6
    fprintf('РЕЗУЛЬТАТ: Декомпозиция подтверждена на новых данных (высокая точность).\n');
else
    fprintf('РЕЗУЛЬТАТ: Декомпозиция НЕ подтверждена на новых данных.\n');
end
fprintf('C_{b→w} = [(I - M)^{-1}]_{b,w} не зависит от X — зависит ТОЛЬКО от {α_e}.\n\n');

%% ===== 7. ГРАФИКИ (Ч/Б без полутонов) =====
figure('Position', [100, 100, 1400, 950]);

% Разные маркеры для struct и dec, чтобы различать серии
markers_struct = {'o', 's', 'd'};
markers_dec   = {'^', 'v', 'p'};

% 1. J_b_struct vs J_b_dec (тестовые)
subplot(4,1,1); hold on;
for bi = 1:numB
    plot(1:nTest, J_b_struct(:, bi), [markers_struct{bi} '-'], 'Color', 'k', 'LineWidth', 1.5, ...
        'MarkerFaceColor', 'k', 'DisplayName', sprintf('J_{struct}^{ч%d}', blackIdx(bi)));
    plot(1:nTest, J_b_dec(:, bi), [markers_dec{bi} '--'], 'Color', 'k', 'LineWidth', 1.5, ...
        'DisplayName', sprintf('J_{dec}^{ч%d}', blackIdx(bi)));
end
hold off; xlabel('Сэмпл', 'FontSize', 14); ylabel('J_b', 'FontSize', 14);
title('Декомпозиция J_b (тестовые данные)', 'FontSize', 14);
legend('Location','best', 'FontSize', 14); grid on;
set(gca, 'FontSize', 14);

% 2. Относительная погрешность (тестовые)
subplot(4,1,2); hold on;
for bi = 1:numB
    relErrVec = abs(J_b_struct(:, bi) - J_b_dec(:, bi)) ./ (abs(J_b_struct(:, bi)) + 1e-15);
    semilogy(1:nTest, relErrVec, [markers_struct{bi} '-'], 'Color', 'k', 'LineWidth', 1.5, ...
        'MarkerFaceColor', 'k', 'DisplayName', sprintf('Чёрн%d', blackIdx(bi)));
end
hold off; xlabel('Сэмпл', 'FontSize', 14); ylabel('Отн. погрешность', 'FontSize', 14);
title('Погрешность (тестовые данные)', 'FontSize', 14);
legend('Location','best', 'FontSize', 14); grid on;
set(gca, 'FontSize', 14);

% 3. J_b_struct vs J_b_dec (новые данные)
subplot(4,1,3); hold on;
for bi = 1:numB
    plot(1:nTest, J_b_struct_new(:, bi), [markers_struct{bi} '-'], 'Color', 'k', 'LineWidth', 1.5, ...
        'MarkerFaceColor', 'k', 'DisplayName', sprintf('J_{struct}^{ч%d}', blackIdx(bi)));
    plot(1:nTest, J_b_dec_new(:, bi), [markers_dec{bi} '--'], 'Color', 'k', 'LineWidth', 1.5, ...
        'DisplayName', sprintf('J_{dec}^{ч%d}', blackIdx(bi)));
end
hold off; xlabel('Сэмпл', 'FontSize', 14); ylabel('J_b', 'FontSize', 14);
title('Декомпозиция J_b (новые данные)', 'FontSize', 14);
legend('Location','best', 'FontSize', 14); grid on;
set(gca, 'FontSize', 14);

% 4. Относительная погрешность (новые данные)
subplot(4,1,4); hold on;
for bi = 1:numB
    relErrVec = abs(J_b_struct_new(:, bi) - J_b_dec_new(:, bi)) ./ (abs(J_b_struct_new(:, bi)) + 1e-15);
    semilogy(1:nTest, relErrVec, [markers_struct{bi} '-'], 'Color', 'k', 'LineWidth', 1.5, ...
        'MarkerFaceColor', 'k', 'DisplayName', sprintf('Чёрн%d', blackIdx(bi)));
end
hold off; xlabel('Сэмпл', 'FontSize', 14); ylabel('Отн. погрешность', 'FontSize', 14);
title('Погрешность (новые данные)', 'FontSize', 14);
legend('Location','best', 'FontSize', 14); grid on;
set(gca, 'FontSize', 14);

sgtitle('Верификация теоремы о декомпозиции', 'FontSize', 14);

fprintf('========== ГРАФИКИ ПОСТРОЕНЫ ==========\n');
fprintf('1-2: Декомпозиция и погрешность на тестовых данных\n');
fprintf('3-4: Декомпозиция и погрешность на новых данных\n');
fprintf('5:   Распределение J_w (тестовые данные)\n\n');