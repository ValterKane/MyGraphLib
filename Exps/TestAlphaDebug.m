%% Тест alpha-производной: проверка механизма поиска dF_source
clear; clc;

import BWGraph.*
import BWGraph.CustomMatrix.*
import BWGraph.RandomGenerator.*

CoreF = coreFunctions.SimpleAddingCoreFunction(1);
betaGen = FullRandomBetaGen(0, 1);

%% DAG B1 -> B2 -> W
B1 = Node(1, 1, 'Black', CoreF, 'linear');
B2 = Node(2, 1, 'Black', CoreF, 'linear');
W  = Node(3, 1, 'White', CoreF, 'linear');

B1.addEdge(B2);
B2.addEdge(W);

NodeWeight = [1, 1, 1];
model = GraphShell(betaGen, NodeWeight, B1, B2, W);

% Фиксируем параметры ПОВЕРХ авто-генерации
B1.getOutEdges().Alfa = 0.3;
B1.getOutEdges().Beta = 0.5;
B2.getOutEdges().Alfa = 0.7;
B2.getOutEdges().Beta = 0.2;
for i = 1:3, model.ListOfNodes(i).Gamma = 1.0; end

% Проверка: один ли и тот же объект?
fprintf('B2 == model.ListOfNodes(2)? %d\n', B2 == model.ListOfNodes(2));
fprintf('W  == model.ListOfNodes(3)? %d\n', W  == model.ListOfNodes(3));

% Проверка: тот же ли TargetNode?
edge_B1_out = B1.getOutEdges();
fprintf('edge_B1.TargetNode == B2? %d\n', edge_B1_out.TargetNode == B2);
fprintf('edge_B1.TargetNode == model.ListOfNodes(2)? %d\n', edge_B1_out.TargetNode == model.ListOfNodes(2));
fprintf('edge ID = %d\n', edge_B1_out.ID);

% Входные данные
XData = BWMatrix();
XData = XData.addRow(10);
XData = XData.addRow(10);
XData = XData.addRow(10);

% Прямой проход
F = model.GetCurrentResult(XData);
fprintf('\nF: B1=%.6f, B2=%.6f, W=%.6f\n', F(1), F(2), F(3));

% Вручную проверяем computeIncomingAlphaDerivativeDirect
dF_dalpha_out = model.computeOutgoingAlphaDerivativeDirect(1);
fprintf('\ndF1_dalpha_out = %.6f\n', dF_dalpha_out);

dF_dalpha_in = model.computeIncomingAlphaDerivativeDirect(2, edge_B1_out, dF_dalpha_out);
fprintf('dF2_dalpha_in (ручной вызов) = %.6f\n', dF_dalpha_in);

% Ожидаем: (F_1 + 0.3*(-5.6213))/1.7 = (7.3077 - 1.6864)/1.7 = 3.3066
expected = (F(1) + 0.3 * dF_dalpha_out) / 1.7;
fprintf('Ожидаемое значение = %.6f\n', expected);

% Теперь вызываем computeAllDerivativesInOrder
[alpha_derivs, ~, ~] = model.computeAllDerivativesInOrder(XData);
allKeys = keys(alpha_derivs);
fprintf('\nВсе alpha-ключи:\n');
for k = 1:numel(allKeys)
    fprintf('  %s = %.6f\n', allKeys{k}, alpha_derivs(allKeys{k}));
end

% Ищем incoming для node 2
for k = 1:numel(allKeys)
    if contains(allKeys{k}, 'alpha_in')
        val = alpha_derivs(allKeys{k});
        fprintf('\nНайден incoming: %s = %.6f (ожидалось %.6f)\n', allKeys{k}, val, expected);
    end
end
