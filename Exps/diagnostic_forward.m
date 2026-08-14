%% Диагностика: сравнение Forward и градиентов
% Запускать в папке 2026 модели, после ручного запуска ExpHeatRealData до обучения

%% 1. Захватываем один пример данных
example_idx = 1;
X_example = XDataTrain(example_idx);
Y_example = YDataTrain(example_idx);

%% 2. Смотрим Forward
modelShell.Forward(X_example);
F_values = modelShell.GetModelResults();
fprintf('\n=== FORWARD ===\n');
for i = 1:numel(F_values)
    fprintf('F_%d = %.4f\n', i, F_values(i));
end

%% 3. Смотрим параметры рёбер
fprintf('\n=== EDGE PARAMETERS ===\n');
for i = 1:numel(modelShell.ListOfNodes)
    node = modelShell.ListOfNodes(i);
    fprintf('Node %d (Gamma=%.4f):\n', i, node.Gamma);
    edges = node.getOutEdges();
    for j = 1:numel(edges)
        e = edges(j);
        fprintf('  -> Node %d: Alpha=%.6f, Beta=%.6f\n', e.TargetNode.ID, e.Alfa, e.Beta);
    end
end

%% 4. Смотрим производные
[alDer, btDer, gmDer] = modelShell.computeAllDerivativesInOrder(X_example);
fprintf('\n=== DERIVATIVES ===\n');
keys_al = keys(alDer);
for k = 1:numel(keys_al)
    key = keys_al{k};
    fprintf('%s: Alpha=%.6f\n', key, alDer(key));
end
keys_bt = keys(btDer);
for k = 1:numel(keys_bt)
    key = keys_bt{k};
    fprintf('%s: Beta=%.6f\n', key, btDer(key));
end
keys_gm = keys(gmDer);
for k = 1:numel(keys_gm)
    key = keys_gm{k};
    fprintf('%s: Gamma=%.6f\n', key, gmDer(key));
end

%% 5. Смотрим невязки (без обучения — просто compute J)
% Захватываем целевое значение
target = Y_example.getRow(1);
fprintf('\n=== TARGET ===\n');
fprintf('Target (T_res_max) = %.4f\n', target);

% Начальные невязки
error_init = F_values(3) - target;
fprintf('\nInitial error (F_3 - target) = %.4f\n', error_init);
fprintf('MAE = %.4f\n', abs(error_init));
