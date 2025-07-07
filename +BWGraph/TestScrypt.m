import BWGraph.*;
import BWGraph.CustomMatrix.*;
import BWGraph.RandomGenerator.*;


% Общая функция для всех вершин
HeatBC = coreFunctions.HeatTransferBC(320.0, 0.72);
rnd = SimpleRandomGen(0.1);

% Создаем вершины
nodeA = Node(1,30,'White',HeatBC,rnd);
nodeB = Node(1,30,'White',HeatBC,rnd);
nodeC = Node(3,30,'White',HeatBC,rnd);
nodeD = Node(4,30,'White',HeatBC,rnd);

% Добавляем соседей
% Вершина А
nodeA.addEdge(nodeB);
nodeA.addEdge(nodeC);
nodeA.addEdge(nodeD);
% Вершина B
nodeB.addEdge(nodeA);
nodeB.addEdge(nodeC);
nodeB.addEdge(nodeD);
% Вершина С
nodeC.addEdge(nodeA);
nodeC.addEdge(nodeB);
nodeC.addEdge(nodeD);
% Вершина D
nodeD.addEdge(nodeA);
nodeD.addEdge(nodeC);
nodeD.addEdge(nodeB);

% Создаем графовую модель
modelShell = GraphShell(nodeA,nodeB,nodeC,nodeD);

% Создаем входные данные
rowA = BWRow([1100, 1250, 1400]);
rowB = BWRow([1100, 1250, 1400]);
rowC = BWRow([1100, 1250, 1400]);
rowD = BWRow([1100, 1250, 1400]);

XData = BWMatrix(rowA, rowB, rowC, rowD);

modelShell.Forward(XData);

result = modelShell.GetModelResults();