%% Тест гибридной δ-нормировки (SCC)
clear; clc;

import BWGraph.*
import BWGraph.CustomMatrix.*
import BWGraph.RandomGenerator.*
import BWGraph.Trainer.*

CoreF = coreFunctions.SimpleAddingCoreFunction(1);
betaGen = FullRandomBetaGen(0, 1);

%% Тест 1: DAG B1 -> B2 -> W
fprintf('=== ТЕСТ 1: DAG B1->B2->W ===\n');

B1 = Node(1, 1, 'Black', CoreF, 'linear');
B2 = Node(2, 1, 'Black', CoreF, 'linear');
W  = Node(3, 1, 'White', CoreF, 'linear');
B1.addEdge(B2);
B2.addEdge(W);

model = GraphShell(betaGen, [1 1 1], B1, B2, W);
B1.getOutEdges().Alfa = 0.3; B1.getOutEdges().Beta = 0.5;
B2.getOutEdges().Alfa = 0.7; B2.getOutEdges().Beta = 0.2;
for i = 1:3, model.ListOfNodes(i).Gamma = 1.0; end

XData = BWMatrix(); XData = XData.addRow(10); XData = XData.addRow(10); XData = XData.addRow(10);
YData = BWMatrix(); YData = YData.addRow(15);

opts = TrainingOptions("Epoches", 1, "BatchSize", 1, "EnablePlateauEscape", false);
trainer = Trainer(model, opts);
trainer.Train(XData, YData, XData, YData);
fprintf('DAG: обучение без ошибок ✓\n');

%% Тест 2: цикл B1<->B2, B2->W
fprintf('\n=== ТЕСТ 2: цикл B1<->B2, B2->W ===\n');

B1c = Node(1, 1, 'Black', CoreF, 'linear');
B2c = Node(2, 1, 'Black', CoreF, 'linear');
Wc  = Node(3, 1, 'White', CoreF, 'linear');
B1c.addEdge(B2c);
B2c.addEdge(B1c);
B2c.addEdge(Wc);

model2 = GraphShell(betaGen, [1 1 1], B1c, B2c, Wc);
B1c.getOutEdges().Alfa = 0.3;
% Устанавливаем через getEdgeToTarget
B2c.getEdgeToTarget(B1c).Alfa = 0.3; B2c.getEdgeToTarget(B1c).Beta = 0.1;
B2c.getEdgeToTarget(Wc).Alfa  = 0.5; B2c.getEdgeToTarget(Wc).Beta  = 0.2;
for i = 1:3, model2.ListOfNodes(i).Gamma = 1.0; end

opts2 = TrainingOptions("Epoches", 1, "BatchSize", 1, "EnablePlateauEscape", false);
trainer2 = Trainer(model2, opts2);
trainer2.Train(XData, YData, XData, YData);
fprintf('Цикл: обучение без ошибок ✓\n');

fprintf('\n=== ИТОГ: SCC-гибрид работает ===\n');
