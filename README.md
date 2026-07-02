# MyGraphLib — MATLAB-библиотека для моделирования черно-белого графа (Black-White Graph)

Библиотека реализует оболочку **Black-White Graph (BWG)** — графовую GLM-модель над произвольными ядровыми функциями-экспертами. Вершины графа делятся на *белые* (с эталонными данными) и *чёрные* (скрытые переменные). Обучение происходит градиентными методами (ADAM) с автоматическим поиском оптимальной топологии.

---

## Быстрый старт

Минимальный пример: граф из трёх вершин (2 чёрные, 1 белая), ядровая функция — модель нагрева, обучение за 5 строк.

```matlab
import BWGraph.*
import BWGraph.Trainer.*
import BWGraph.RandomGenerator.*

% 1. Ядровые функции
Heat_1 = coreFunctions.Heating2DModel(30, 21, 21, 50, 1.5e-5, 0.3, 0.360, 30, 10);
Heat_2 = coreFunctions.Heating2DModel(60, 21, 21, 60, 1.5e-5, 0.3, 0.360, 60, 10);
Heat_w = coreFunctions.Heating2DModel(90, 21, 21, 70, 1.5e-5, 0.3, 0.360, 70, 10);

% 2. Генератор β (α генерируется автоматически с учётом топологии)
betaGen = FullRandomBetaGen(1, 100);

% 3. Вершины и связи
node1 = Node(1, 30, 'Black', Heat_1);
node2 = Node(2, 30, 'Black', Heat_2);
nodeW = Node(3, 30, 'White', Heat_w);
node1.addEdge(node2);
node2.addEdge(nodeW);

% 4. Оболочка графа
model = GraphShell(betaGen, [1 0.5 0.5], node1, node2, nodeW);
model.DrawGraph_New('Модель нагрева');

% 5. Обучение
opts = TrainingOptions("Epoches", 200, "LearningRate", 0.01, ...
    "AutoCalibrateClip", true, "ErrorMetric", "mae", "LossFunction", "mae");
trainer = Trainer(model, opts);
trainer.Train(X_train, Y_train, X_test, Y_test);
```

> **Важно:** α-параметры теперь генерируются **автоматически внутри `GraphShell`** через `GenerateTopologyAwareAlpha()` — с гарантией выполнения условия устойчивости: **Σα_in(v) < 1 + Σα_out(v)** для каждой вершины v. Внешний `AlphaGenerator` больше не требуется.

---

## Структура проекта

```
MyGraphLib/
│
├── +BWGraph/                              # Пространство имён модели
│   ├── Edge.m                             # Ребро графа (α, β)
│   ├── GraphShell.m                       # Оболочка графа (прямой проход, производные, топология)
│   ├── Node.m                             # Вершина графа
│   ├── NodeColor.m                        # Перечисление: White / Black
│   │
│   ├── +CustomMatrix/                     # Jagged-матрица (вектора переменной длины)
│   │   ├── BWMatrix.m                     # Матрица: XData / YData
│   │   └── BWRow.m                        # Строка матрицы
│   │
│   ├── +RandomGenerator/                  # Генераторы начальных весов
│   │   ├── IRandomGen.m                   # Интерфейс
│   │   ├── FullRandomGen.m                # Базовый Uniform(min, max)
│   │   ├── FullRandomAlfaGen.m            # Генератор α (legacy, более не используется)
│   │   ├── FullRandomBetaGen.m            # Генератор β
│   │   ├── HybridAlphaGenerator.m         # Гибридный α-генератор (legacy)
│   │   └── HybridBetaGenerator.m          # Гибридный β-генератор
│   │
│   └── +Trainer/                          # Обучение
│       ├── Trainer.m                      # Алгоритмы: градиентный спуск, плато, структурный поиск
│       └── TrainingOptions.m              # Гиперпараметры обучения
│
├── +coreFunctions/                        # Ядровые функции (ICoreF / ITunableCoreF)
│   ├── ICoreF.m                           # Интерфейс ядровой функции
│   ├── ITunableCoreF.m                    # Интерфейс настраиваемой функции
│   ├── Heating2DModel.m                   # Модель нестационарной теплопроводности (2D)
│   ├── Heating2DTunableModel.m            # Настраиваемая модель нагрева
│   ├── Heating2DWithRolling.m             # Модель нагрева с прокаткой
│   ├── PlateHeatingModel.m                # Пластина
│   ├── LinearFunction.m                   # Линейная: y = ax + b
│   ├── LinearRegression.m                 # Линейная регрессия
│   ├── SigmoidFunction.m                  # Сигмоида
│   ├── HeatTransferBC.m                   # Граничные условия теплообмена
│   ├── HeatLinearRegression.m             # Линейная регрессия + тепло
│   └── SimpleAddingCoreFunction.m         # Простое сложение входов
│
├── Exps/                                  # Эксперименты (~15 скриптов)
│   ├── ExpHeat.m                          # Базовая модель нагрева
│   ├── ExpHeatRealData.m                  # Нагрев на реальных данных
│   ├── ExpHeatRealDataTunable.m           # Настраиваемая модель + реальные данные
│   ├── ExpHeatRealData_Structural.m       # Структурный поиск на реальных данных
│   ├── ExpHeatOldData.m                   # Старые данные нагрева
│   ├── ExpHeatWith2Dim.m                  # Двумерный вход
│   ├── ExpLinear.m, ExpLinearTwoX.m       # Линейные эксперименты
│   ├── ExpSigmoid.m                       # Сигмоидные ядра
│   ├── ExpSimpleData.m                    # Простые данные
│   ├── ExpTunableVerification.m           # Верификация настраиваемых функций
│   ├── ExpVerificationTheorem.m           # Численная проверка теоремы о декомпозиции
│   ├── ExpStructuralSearchTest.m          # Тест жадного NAS
│   ├── ExpScrypt.m                        # Вспомогательный скрипт
│   └── NonlinearExps/FirstExp.m           # Нелинейный эксперимент
│
├── README.md
└── Spec.md
```

---

## Основные концепции

### Вершины (`Node`)

| Свойство | Тип | Описание |
|---|---|---|
| `NodeType` | `NodeColor.White` / `NodeColor.Black` | Белая — есть эталон, чёрная — скрытая |
| `NodeFunction` | `ICoreF` | Ядровая функция (может быть `[]` для чёрных вершин) |
| `ActivationType` | `"linear"` / `"sigmoid"` / `"relu"` / `"tanh"` | Функция активации после `γ·CoreFunction` |
| `Gamma` | `double` | Мультипликативный вес ядра |
| `FResult` | `double` | Текущее значение в вершине (вычисляется `Forward`-проходом) |

Формула вершины (до применения активации):

$$F_v = \sigma\!\left(\gamma_v \cdot f_v(x_v) \;+\!\! \sum_{u \in \text{In}(v)} \bigl(\alpha_{u \to v} \cdot F_u + \beta_{u \to v}\bigr)\right)$$

где σ — функция активации, f_v — ядровая функция, In(v) — входящие в v рёбра.

### Рёбра (`Edge`)

Каждое ребро `u → v` несёт два параметра:

| Параметр | Роль |
|---|---|
| **α** (Alfa) | Линейный коэффициент передачи: вклад `F_u` в `F_v` умножается на α |
| **β** (Beta) | Аддитивное смещение |

Генерация α теперь **внутренняя**: метод `GenerateTopologyAwareAlpha()` распределяет α так, чтобы ∀v выполнялось **условие устойчивости**:

$$\sum_{e \in \text{In}(v)} \alpha_e \;<\; 1 + \sum_{e \in \text{Out}(v)} \alpha_e$$

### Прямой проход (`Forward`)

Система уравнений для всех N вершин:

$$F_v = \frac{L_v + G_{\text{in}}(v) - \sum_{e \in \text{Out}(v)} \beta_e}{1 + \sum_{e \in \text{Out}(v)} \alpha_e}, \quad v = 1,\dots,N$$

где L_v = γ_v · f_v(x_v), G_in(v) = Σ(α_e·F_u + β_e) — вклад входящих рёбер.

**Матричная форма:** **(D − A_in) · Φ = L + B_in − B_out**

- D — диагональная: D_{ii} = 1 + Σα_out(i)
- A_in — матрица входящих α: (A_in)_{ij} = α_{j→i}
- B_in, B_out — вектора сумм β по входящим/исходящим рёбрам

**Кеширование:** M⁻¹ = (D − A_in)⁻¹ и const = B_in − B_out не зависят от входных данных x — вычисляются **один раз** при изменении параметров рёбер и переиспользуются между сэмплами. Ускорение ~10×.
- **Кеширование `CalcCoreFunction`:** результат f_v(x_v) кешируется в `Node` между вызовами с одинаковыми входными данными.

---

## Данные: формат `BWMatrix`

Модель работает с *jagged matrices* — строки могут иметь разную длину (разные вершины принимают разное количество параметров).

```matlab
% Создание одного сэмпла
sample = BWMatrix();
sample = sample.addRow([t1; Tinf1]);  % Вход вершины 1
sample = sample.addRow([t2; Tinf2]);  % Вход вершины 2
sample = sample.addRow([t3; Tinf3]);  % Вход вершины 3

% Массив сэмплов
XData = repmat(BWMatrix(), N, 1);
for i = 1:N
    XData(i) = XData(i).addRow(params_for_v1(i,:));
    % ...
end
```

**Порядок строк в `BWMatrix` соответствует порядку вершин в `GraphShell.ListOfNodes`.**

### Загрузка из Excel с кириллическими заголовками

```matlab
data = readtable("file.xlsx", VariableNamingRule="preserve");
```

---

## Обучение: `Trainer` + `TrainingOptions`

### Полный список параметров `TrainingOptions`

| Параметр | По умолчанию | Описание |
|---|---|---|
| **Основные** |||
| `LearningRate` | 0.001 | Начальная скорость обучения |
| `Epoches` | 100 | Максимальное число эпох |
| `BatchSize` | 1 | Размер батча |
| `TargetError` | 1e-5 | Целевая ошибка (ранняя остановка) |
| **ADAM** |||
| `Beta1` | 0.9 | Затухание первого момента |
| `Beta2` | 0.999 | Затухание второго момента |
| `Eps` | 1e-8 | Численная стабильность |
| **Клиппинг градиентов** |||
| `ClipUp` / `ClipDown` | 1e5 / −1e5 | Общие границы |
| `ClipUp_Alpha` / `ClipDown_Alpha` | `[]` | Раздельный клиппинг для α |
| `ClipUp_Beta` / `ClipDown_Beta` | `[]` | Раздельный клиппинг для β |
| `ClipUp_Gamma` / `ClipDown_Gamma` | `[]` | Раздельный клиппинг для γ |
| `AutoCalibrateClip` | `false` | Автокалибровка границ по первому батчу |
| `ClipPercentile` | 95 | Перцентиль для автокалибровки |
| **Регуляризация** |||
| `Lambda_Alph` | 0.01 | L2-регуляризация α |
| `Lambda_Beta` | 0.01 | L2-регуляризация β |
| `Lambda_Gamma` | 0.01 | L2-регуляризация γ |
| `Lambda_Agg` | 0 | Штраф агрегации ошибок |
| `Lambda_Self` | 0 | Собственная регуляризация |
| `Lambda_Struct` | 0 | Структурная регуляризация |
| `Lambda_Stability` | 0 | Штраф нарушения условия устойчивости |
| **Метрики** |||
| `ErrorMetric` | `"mae"` | `mae`, `mse`, `rmse`, `mape` |
| `LossFunction` | `"mae"` | `mae`, `mse`, `huber`, `logcosh` |
| `TargetNodeIndices` | `[]` | Индексы белых вершин для ошибки (`[]` = все) |
| `HuberDelta` | 1 | Параметр δ для Huber Loss |
| **Выход из плато (Алгоритм 1)** |||
| `EnablePlateauEscape` | `true` | Включает RpShift-оператор при застревании |
| `RpShiftPercent` | 20 | Процент случайного смещения параметров |
| `LRDecayInterval` | 1 | Каждые N эпох: η ← η_init / √epoch |
| **Структурный поиск / NAS (Алгоритм 2)** |||
| `EnableStructuralSearch` | `false` | Включить жадный поиск топологии |
| `StructuralSearchInterval` | 50 | Каждые N эпох — шаг поиска |
| `StructuralSearchCandidates` | 10 | Число кандидатов-рёбер на проверку |
| `StructuralSearchEpochs` | 5 | Эпох быстрой настройки кандидата |
| `StructuralSearchMaxEdges` | `inf` | Максимальное число рёбер |
| `StructuralSearchMinEdges` | 0 | Минимальное число рёбер |
| `StructuralCooldown` | 2 | Эпох охлаждения после изменения топологии |
| `StructuralComplexityPenalty` | 0 | L1-штраф за каждое ребро |
| `StructuralCleanupThreshold` | 0.01 | Порог α для удаления «мусорных» рёбер |
| **Инициализация α** |||
| `AlphaMin` | 0.01 | Минимальное значение α |
| `AlphaSafetyFactor` | 0.8 | Доля бюджета D(v) для Σα_in |
| `StructInitAlpha` | 0.5 | Начальное α для новых рёбер |
| `StructInitBeta` | 1 | Начальное β для новых рёбер |

### Алгоритм 1: Выход из плато (Plateau Escape)

При застревании (отсутствие улучшения `p_e` эпох подряд):
1. К параметрам применяется случайное смещение (`RpShiftPercent`% от текущих значений)
2. LR сбрасывается до начального
3. Максимум `p_p` попыток, затем остановка

LR-шедулинг: **η_epoch = η_init / √epoch** — затухание по корню эпохи для стабилизации сходимости.

### Алгоритм 2: Структурный поиск (Greedy NAS)

Каждые `StructuralSearchInterval` эпох:
1. Генерируются кандидаты — отсутствующие рёбра между всеми вершинами
2. Каждый кандидат быстро настраивается (`StructuralSearchEpochs` эпох)
3. Лучший кандидат (по ошибке на тесте) добавляется в граф
4. Глобальный кеш (`globalEdgeCache`) исключает повторную проверку отклонённых рёбер
5. После конвергенции — `CleanupRedundantEdges()` удаляет рёбра с α ниже порога

### Визуализация обучения

Дашборд 3×3 в реальном времени:
- Ошибки train/test (с маркерами структурных изменений)
- Learning rate
- Время эпохи
- Разница ошибок (early stopping)
- Матрица смежности (текущая топология)
- Визуальный граф (α, β, γ на рёбрах)

---

## API: `GraphShell` — методы манипуляции топологией

Методы для программного добавления/удаления рёбер (используются структурным поиском):

```matlab
model.addEdgeBetween(srcIdx, dstIdx, alpha, beta)   % Добавить ребро
model.removeEdgeBetween(srcIdx, dstIdx)              % Удалить ребро
flag = model.hasEdge(srcIdx, dstIdx)                 % Проверить существование
n = model.numEdges()                                 % Количество рёбер
A = model.getAdjacencyMatrix()                       % Матрица смежности
model.GenerateTopologyAwareAlpha()                   % Перегенерировать α под новую топологию
model.invalidateForwardCache()                       % Сбросить кеш прямого прохода
```

---

## API: полный справочник методов

### `GraphShell`

| Метод | Описание |
|---|---|
| `GraphShell(BetaGenerator, NodeWeight, ...Node)` | Конструктор. β-генератор, веса, вершины |
| `LoadFromFile(filename)` *(static)* | Загрузить граф из .mat |
| `Forward(Data)` | Прямой проход, обновляет `FResult` |
| `GetCurrentResult(XData)` | Прямой проход → вектор результатов |
| `GetModelResults()` | Текущие `FResult` всех вершин |
| `GetNumOfWhiteNode()` / `GetNumOfBlackNode()` | Количество белых/чёрных |
| `GetWhiteNodes()` / `GetBlackNodes()` | Массивы вершин |
| `GetWhiteNodesIndices()` / `GetBlackNodesIndices()` | Индексы |
| `IsWhiteVertice(idx)` / `IsBlackVertice(idx)` | Проверка типа |
| `getIncomingEdges(node)` / `getIncomingNeighbors(node)` | Входящие связи |
| `buildSystemMatrices(Data)` | D, A_in, B_in, B_out, L |
| `computeAllDerivativesInOrder(XData)` | Все ∂/∂α, ∂/∂β, ∂/∂γ в топологическом порядке |
| `computeGammaDerivativeForNode(idx, data)` | ∂F/∂γ для вершины |
| `computeOutgoingAlphaDerivativeForEdge(idx)` | ∂F/∂α исходящего ребра |
| `computeOutgoingBetaDerivativeForEdge(idx)` | ∂F/∂β исходящего ребра |
| `DrawGraph_New(titleStr, ax)` | Визуализация графа (опционально на заданных осях) |
| `GenerateTopologyAwareAlpha()` | Генерация α с гарантией устойчивости |
| `invalidateForwardCache()` | Сброс кеша M⁻¹ |
| `addEdgeBetween(src, dst, α, β)` | Добавить направленное ребро |
| `removeEdgeBetween(src, dst)` | Удалить направленное ребро |
| `hasEdge(src, dst)` | Проверить существование ребра |
| `numEdges()` | Количество рёбер |
| `getAdjacencyMatrix()` | Матрица смежности N×N |

### `Trainer`

| Метод | Описание |
|---|---|
| `Trainer(Graph, TrainingOptions)` | Конструктор |
| `Train(X_train, Y_train, X_test, Y_test)` | Обучение (основной цикл) |
| `GetGraph()` | Вернуть граф |
| `SaveBestParameters()` | Сохранить лучшие параметры |
| `RestoreBestParameters()` | Восстановить лучшие параметры |
| `RandomShiftParameters()` | RpShift-оператор (выход из плато) |
| `CalculateError(X, Y, indices, metric)` | Вычислить ошибку |
| `ResetTrainingState()` | Сброс состояния (новый запуск) |
| `StructuralSearchStep(...)` | Один шаг жадного NAS |
| `CleanupRedundantEdges(...)` | Зачистка рёбер после конвергенции |
| `UpdateStructuralPlot(ax)` | Обновить матрицу смежности на дашборде |

### `Node`

| Метод | Описание |
|---|---|
| `Node(ID, initVal, nodeType, nodeFunction)` | Конструктор |
| `addEdge(targetNode)` | Добавить исходящее ребро |
| `removeEdgeByTarget(targetNode)` | Удалить ребро |
| `getEdgeToTarget(targetNode)` | Получить ребро |
| `getOutEdges()` / `getOutEdgesMap()` | Исходящие рёбра |
| `getNeighbors()` | Соседи |
| `calcNodeFunc(inputData)` | γ·CoreFunction → activation |
| `calcRawCoreFunction(inputData)` | CoreFunction (с кешированием) |
| `getFResult()` / `setFResult(v)` | Текущее значение |
| `getNodeType()` | `White` / `Black` |
| `getNodeFunction()` | Ядровая функция |
| `getActivationType()` | Тип активации |

### `Edge`

| Свойство | Описание |
|---|---|
| `Alfa` | Линейный коэффициент α |
| `Beta` | Аддитивное смещение β |
| `SourceNode` / `TargetNode` | Вершины |
| `ID` | Идентификатор |

---

## Эксперименты

Все эксперименты находятся в `Exps/`. Для запуска:

```matlab
cd MyGraphLib
run("Exps/ExpHeatRealData.m")
```

Базовые сценарии:
- **ExpHeat / ExpHeatRealData** — модель нагрева, синтетика / реальные данные
- **ExpHeatRealData_Structural** — структурный поиск на реальных данных
- **ExpStructuralSearchTest** — тест жадного NAS на синтетике
- **ExpVerificationTheorem** — численная проверка теоремы о декомпозиции C_{b→w}
- **ExpHeatWith2Dim** — двумерный вход (t, Tinf)
- **ExpLinear / ExpLinearTwoX** — линейные ядра
- **ExpSigmoid** — сигмоидная активация
- **ExpTunableVerification** — настраиваемые ядровые функции (ITunableCoreF)

---

## Технические аспекты

**Реализовано:**
- Мультипликативные (γ) и аддитивные (β) зависимости между ядрами
- Линейные коэффициенты передачи (α) с гарантией устойчивости
- Нелинейные активации: `linear`, `sigmoid`, `relu`, `tanh`
- ADAM-оптимизатор с раздельным клиппингом и автокалибровкой
- Механизм выхода из плато (RpShift)
- Жадный структурный поиск (NAS) с глобальным кешем
- Кеширование прямого прохода и ядровых функций
- Сохранение/загрузка модели через `LoadFromFile`

**В плане:**
- Векторизация внутренних циклов
- Поддержка GPU (через `gpuArray`)
- Поддержка batch-режима (BatchSize > 1)
