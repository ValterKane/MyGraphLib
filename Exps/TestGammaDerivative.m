%% Тест gamma-производной: аналитика vs конечные разности
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

model = GraphShell(betaGen, [1 1 1], B1, B2, W);
B1.getOutEdges().Alfa = 0.3; B1.getOutEdges().Beta = 0.5;
B2.getOutEdges().Alfa = 0.7; B2.getOutEdges().Beta = 0.2;

gammas = [0.5, 0.8, 1.2];
for i = 1:3, model.ListOfNodes(i).Gamma = gammas(i); end

XData = BWMatrix(); XData = XData.addRow(10); XData = XData.addRow(10); XData = XData.addRow(10);
F = model.GetCurrentResult(XData);
fprintf('F: B1=%.6f, B2=%.6f, W=%.6f\n\n', F(1), F(2), F(3));

[~, ~, gamma_derivs] = model.computeAllDerivativesInOrder(XData);
keys_g = keys(gamma_derivs);

eps = 1e-6;
for k = 1:numel(keys_g)
    key = keys_g{k};
    analytic = gamma_derivs(key);
    nodeIdx = sscanf(key, 'node%d_gamma');

    orig_gamma = model.ListOfNodes(nodeIdx).Gamma;
    model.ListOfNodes(nodeIdx).Gamma = orig_gamma + eps;
    F_plus = model.GetCurrentResult(XData);
    model.ListOfNodes(nodeIdx).Gamma = orig_gamma - eps;
    F_minus = model.GetCurrentResult(XData);
    model.ListOfNodes(nodeIdx).Gamma = orig_gamma;

    fd = (F_plus(nodeIdx) - F_minus(nodeIdx)) / (2*eps);

    fprintf('%s: анал=%.8f  FD=%.8f  diff=%.2e  %s\n', ...
        key, analytic, fd, abs(analytic - fd), ...
        iif(abs(analytic - fd) < 1e-5, 'OK', 'ОШИБКА'));
end

%% Проверка без NodeFunction
fprintf('\n=== Чёрная вершина БЕЗ NodeFunction ===\n');
B3 = Node(4, 1, 'Black', [], 'linear');
B3.addEdge(W);

model2 = GraphShell(betaGen, [1 1 1 1], B1, B2, B3, W);
for i = 1:3, model2.ListOfNodes(i).Gamma = gammas(i); end
model2.ListOfNodes(4).Gamma = 0.5;

XData4 = BWMatrix(); XData4 = XData4.addRow(10); XData4 = XData4.addRow(10); XData4 = XData4.addRow(10); XData4 = XData4.addRow(10);
F4 = model2.GetCurrentResult(XData4);
fprintf('F: B1=%.3f, B2=%.3f, B3=%.3f, W=%.3f\n', F4(1), F4(2), F4(3), F4(4));

[~, ~, gamma_derivs2] = model2.computeAllDerivativesInOrder(XData4);
keys_g2 = keys(gamma_derivs2);
fprintf('Ключи gamma: ');
for k = 1:numel(keys_g2), fprintf('%s=%.6f  ', keys_g2{k}, gamma_derivs2(keys_g2{k})); end
fprintf('\n');

% B3 без NodeFunction: f(x)=0 → L=γ·0=0 → ∂F/∂γ=0/D=0
% Ожидаем: node4_gamma = 0
fprintf('Ожидаем node4_gamma=0 (нет NodeFunction → нет градиента по gamma)\n');

function s = iif(cond, t, f)
    if cond, s = t; else, s = f; end
end
