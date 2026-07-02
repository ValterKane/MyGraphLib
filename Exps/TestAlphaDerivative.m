%% Тест alpha-производной: аналитика vs конечные разности
clear; clc;

import BWGraph.*
import BWGraph.CustomMatrix.*
import BWGraph.RandomGenerator.*

% Ядро: f(x) = x (простое сложение одного параметра)
CoreF = coreFunctions.SimpleAddingCoreFunction(1);

betaGen = FullRandomBetaGen(0, 1);

%% Тест 1: DAG из трёх вершин B1 -> B2 -> W
fprintf('=== ТЕСТ 1: DAG B1 -> B2 -> W ===\n');

B1 = Node(1, 1, 'Black', CoreF, 'linear');
B2 = Node(2, 1, 'Black', CoreF, 'linear');
W  = Node(3, 1, 'White', CoreF, 'linear');

B1.addEdge(B2);
B2.addEdge(W);

NodeWeight = [1, 1, 1];
model = GraphShell(betaGen, NodeWeight, B1, B2, W);

% Фиксируем параметры
B1.getOutEdges().Alfa = 0.3;
B1.getOutEdges().Beta = 0.5;
B2.getOutEdges().Alfa = 0.7;
B2.getOutEdges().Beta = 0.2;
gammas = [1.0, 1.0, 1.0];
for i = 1:3, model.ListOfNodes(i).Gamma = gammas(i); end

% Входные данные: x=10 для всех вершин
XData = BWMatrix();
XData = XData.addRow(10);
XData = XData.addRow(10);
XData = XData.addRow(10);

% Прямой проход
F = model.GetCurrentResult(XData);
fprintf('F: B1=%.6f, B2=%.6f, W=%.6f\n', F(1), F(2), F(3));

% Аналитические производные
[alpha_derivs, beta_derivs, gamma_derivs] = model.computeAllDerivativesInOrder(XData);

% Печатаем все ключи для отладки
fprintf('\nКлючи alpha_derivatives:\n');
allKeys = keys(alpha_derivs);
for k = 1:numel(allKeys)
    fprintf('  %s = %.6f\n', allKeys{k}, alpha_derivs(allKeys{k}));
end

eps = 1e-6;

%% Проверка ∂F_2/∂α_{B1→B2} (incoming для B2)
% Ищем ключ, содержащий '_alpha_in'
key = '';
for k = 1:numel(allKeys)
    if contains(allKeys{k}, 'alpha_in')
        key = allKeys{k};
        break;
    end
end
analytic = alpha_derivs(key);
fprintf('\nКлюч incoming alpha: %s\n', key);
fprintf('\n∂F_2/∂α_{1→2}:\n');
fprintf('  Аналитика:  %.8f\n', analytic);

% Конечная разность
orig_alpha = B1.getOutEdges().Alfa;
B1.getOutEdges().Alfa = orig_alpha + eps;
F_plus = model.GetCurrentResult(XData);
B1.getOutEdges().Alfa = orig_alpha - eps;
F_minus = model.GetCurrentResult(XData);
B1.getOutEdges().Alfa = orig_alpha;
fd = (F_plus(2) - F_minus(2)) / (2*eps);
fprintf('  Конеч.разн: %.8f\n', fd);
fprintf('  Разница:    %.2e  %s\n', abs(analytic - fd), ...
    iif(abs(analytic - fd) < 1e-5, '✓ OK', '✗ ОШИБКА'));

%% Проверка ∂F_1/∂α_{B1→B2} (outgoing для B1)
key_out = '';
for k = 1:numel(allKeys)
    if contains(allKeys{k}, 'alpha_out')
        key_out = allKeys{k};
        break;
    end
end
analytic = alpha_derivs(key_out);
fprintf('\nКлюч outgoing alpha: %s\n', key_out);
fprintf('\n∂F_1/∂α_{1→2} (outgoing):\n');
fprintf('  Аналитика:  %.8f\n', analytic);

B1.getOutEdges().Alfa = orig_alpha + eps;
F_plus = model.GetCurrentResult(XData);
B1.getOutEdges().Alfa = orig_alpha - eps;
F_minus = model.GetCurrentResult(XData);
B1.getOutEdges().Alfa = orig_alpha;
fd = (F_plus(1) - F_minus(1)) / (2*eps);
fprintf('  Конеч.разн: %.8f\n', fd);
fprintf('  Разница:    %.2e  %s\n', abs(analytic - fd), ...
    iif(abs(analytic - fd) < 1e-5, '✓ OK', '✗ ОШИБКА'));

%% Проверка ∂F_W/∂α_{B2→W} (incoming для W)

% Ищем второй incoming-ключ (для W)
keys_in = {};
for k = 1:numel(allKeys)
    if contains(allKeys{k}, 'alpha_in')
        keys_in{end+1} = allKeys{k};
    end
end
key_w_in = keys_in{end};  % последний incoming — от B2→W
analytic = alpha_derivs(key_w_in);
fprintf('\nКлюч incoming alpha (W): %s\n', key_w_in);
fprintf('\n∂F_3/∂α_{2→3}:\n');
fprintf('  Аналитика:  %.8f\n', analytic);

orig_alpha2 = B2.getOutEdges().Alfa;
B2.getOutEdges().Alfa = orig_alpha2 + eps;
F_plus = model.GetCurrentResult(XData);
B2.getOutEdges().Alfa = orig_alpha2 - eps;
F_minus = model.GetCurrentResult(XData);
B2.getOutEdges().Alfa = orig_alpha2;
fd = (F_plus(3) - F_minus(3)) / (2*eps);
fprintf('  Конеч.разн: %.8f\n', fd);
fprintf('  Разница:    %.2e  %s\n', abs(analytic - fd), ...
    iif(abs(analytic - fd) < 1e-5, '✓ OK', '✗ ОШИБКА'));

%% Проверка ∂F_W/∂α_{B1→B2} — полная цепочечная производная
fprintf('\n=== ЦЕПОЧЕЧНАЯ ПРОИЗВОДНАЯ ∂F_W/∂α_{1→2} ===\n');
fprintf('(W не имеет прямого ключа для α_{1→2} — только через цепочку)\n');

% Аналитически через цепное правило (золотой стандарт)
dF2_da12 = analytic;  % уже проверенная ∂F_2/∂α_{1→2} (incoming)
% ∂F_W/∂F_2 = α_{2→W} / D(W)  (нет исходящих из W → D=1)
% ∂F_W/∂F_2 ещё можно трактовать как:
% F_W = L_W + α_{2W}·F_2 + β_{2W}
dFW_dF2 = B2.getOutEdges().Alfa; % = α_{2→3}, т.к. D(W)=1
golden = dFW_dF2 * dF2_da12;
fprintf('  Цепное правило (золотой стандарт): %.8f\n', golden);

% Конечная разность
B1.getOutEdges().Alfa = orig_alpha + eps;
F_plus = model.GetCurrentResult(XData);
B1.getOutEdges().Alfa = orig_alpha - eps;
F_minus = model.GetCurrentResult(XData);
B1.getOutEdges().Alfa = orig_alpha;
fd = (F_plus(3) - F_minus(3)) / (2*eps);
fprintf('  Конечная разность:                 %.8f\n', fd);
fprintf('  Цепное vs FD разница:              %.2e\n', abs(golden - fd));

% Теперь: что даёт код через J_total-механизм?
% J_total(2) = λ_struct * (δ_out(2→3)·J(3) + ...)
% δ_out(2→3) = α_23 / D(2) = 0.7 / 1.7 ≈ 0.4118
% Если J(3)=1 (единичная ошибка в W):
delta_23 = B2.getOutEdges().Alfa / (1 + B2.getOutEdges().Alfa);
fprintf('\n  δ_out(2→3) = %.6f\n', delta_23);

% Вклад через outgoing B1 + incoming B2 при J(3)=1, J(2)=δ·J(3):
% J(2) ≈ delta_23 * 1  (только структурная часть, λ_self=0 для простоты)
% code_gradient = J(1)*dF1_da + J(2)*dF2_da_in
dF1_da = alpha_derivs(key_out);  % ∂F_1/∂α_out уже проверена выше
code_indirect = delta_23 * dF2_da12;  % вклад через B2 (J(2)*dF2/da)
fprintf('  Вклад через J(2)*∂F_2/∂α_in = %.8f\n', code_indirect);
fprintf('  Золотой стандарт (J(3)=1)    = %.8f\n', golden);

% Сравнение: code_indirect vs golden
% code: δ_23 · dF2/da = α/(1+α)·F1/((1+α12)·(1+α23))
% golden: α · dF2/da = α · F1/((1+α12)·(1+α23))
ratio = golden / code_indirect;
fprintf('  Отношение golden/code = %.4f  (ожидаем D(2)=%.4f)\n', ratio, 1+B2.getOutEdges().Alfa);

%% Итог
fprintf('\n=== ИТОГ ===\n');
fprintf('Все прямые производные (incoming/outgoing) проверены через конечные разности.\n');
fprintf('Цепочечная ∂F_W/∂α_{1→2} через δ-пропагацию даёт занижение в D(2)=%.4f раз.\n', 1+B2.getOutEdges().Alfa);
fprintf('Это означает, что J_total-пропагация НЕ компенсирует отсутствие\n');
fprintf('прямого ∂F_W/∂α_{1→2} в computeAllDerivativesInOrder.\n');

function s = iif(cond, t, f)
    if cond, s = t; else, s = f; end
end
