# MyGraphLib — MATLAB-библиотека для моделирования черно-белого графа (Black-White Graph)

Библиотека реализует оболочку **Black-White Graph (BWG)** — графовую GLM-модель над произвольными ядровыми функциями-экспертами. Вершины графа делятся на *белые* (с эталонными данными) и *чёрные* (скрытые переменные). Обучение — градиентными методами (ADAM) с автоматическим поиском оптимальной топологии. Поддерживается многоэтапный прямой проход (K > 1) с обратным распространением через этапы (BPTT) и векторным контекстом между этапами.

---

## Навигация

- [Быстрый старт](#быстрый-старт)
- [Структура проекта](#структура-проекта)
- [Основные концепции](#основные-концепции)
  - [Вершины (Node)](#вершины-node)
  - [Рёбра (Edge)](#рёбра-edge)
  - [Прямой проход (Forward)](#прямой-проход-forward)
  - [Многоэтапный прямой проход (K > 1)](#многоэтапный-прямой-проход-k--1)
  - [Обратный проход: функция потерь чёрной вершины](#обратный-проход-функция-потерь-чёрной-вершины)
  - [BPTT: обратное распространение через этапы](#bptt-обратное-распространение-через-этапы)
  - [Устойчивость](#устойчивость)
- [Векторный контекст и ContextProjector](#векторный-контекст-и-contextprojector)
- [Данные: формат BWMatrix](#данные-формат-bwmatrix)
- [Обучение: Trainer + TrainingOptions](#обучение-trainer--trainingoptions)
- [API: GraphShell](#api-graphshell---методы-манипуляции-топологией)
- [API: полный справочник методов](#api-полный-справочник-методов)
- [Эксперименты](#эксперименты)
- [Результаты](#результаты)
- [Технические аспекты](#технические-аспекты)

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

% 4. Оболочка графа (α = 0; генерируется в Trainer через GenerateTopologyAwareAlpha)
model = GraphShell(betaGen, [1 0.5 0.5], node1, node2, nodeW);
model.DrawGraph_New('Модель нагрева');

% 5. Обучение
opts = TrainingOptions("Epoches", 200, "LearningRate", 0.01, ...
    "AutoCalibrateClip", true, "ErrorMetric", "mae", "LossFunction", "mae");
trainer = Trainer(model, opts);
trainer.Train(X_train, Y_train, X_test, Y_test);
```

> **Важно:** α генерируется через публичный метод `GenerateTopologyAwareAlpha(SafetyFactor)`, вызываемый **из конструктора `Trainer`**. Это гарантирует условие устойчивости: **Σα_in(v) < 1 + Σα_out(v)** для каждой вершины v. При использовании `GraphShell` без `Trainer` метод нужно вызвать вручную. Параметр `SafetyFactor` (по умолчанию 0.8) управляется через `TrainingOptions.AlphaSafetyFactor`.

---

## Структура проекта

```
MyGraphLib/
│
├── +BWGraph/                              # Пространство имён модели
│   ├── Edge.m                             # Ребро графа (α, β)
│   ├── GraphShell.m                       # Оболочка графа (Forward, BackpropContext, производные, топология)
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
│       ├── Trainer.m                      # ADAM, BPTT, выход из плато, структурный поиск
│       └── TrainingOptions.m              # Гиперпараметры обучения
│
├── +coreFunctions/                        # Ядровые функции (ICoreF / ITunableCoreF)
│   ├── ICoreF.m                           # Интерфейс ядровой функции (+ контекст: SupportsContext, AugmentInput, CalcContextDerivative)
│   ├── ITunableCoreF.m                    # Интерфейс настраиваемой функции
│   ├── ContextProjector.m                 # Проектор контекста (без собственного решателя, для multi-stage цепочек)
│   ├── Heating2DModel.m                   # Модель нестационарной теплопроводности (2D) с поддержкой контекста
│   ├── Heating2DTunableModel.m            # Настраиваемая модель нагрева
│   ├── Heating2DWithRolling.m             # Модель нагрева с прокаткой
│   ├── LinearFunction.m                   # Линейная: y = ax + b
│   ├── LinearRegression.m                 # Линейная регрессия
│   ├── SigmoidFunction.m                  # Сигмоида
│   ├── HeatTransferBC.m                   # Граничные условия теплообмена
│   ├── HeatLinearRegression.m             # Линейная регрессия + тепло
│   └── SimpleAddingCoreFunction.m         # Простое сложение входов
│
├── Exps/                                  # Эксперименты
│   ├── ExpHeat.m                          # Базовая модель нагрева
│   ├── ExpHeatRealData.m                  # Нагрев на реальных данных
│   ├── ExpHeatRealDataTunable.m           # Настраиваемая модель + реальные данные
│   ├── ExpHeatRealData_Structural.m       # Структурный поиск на реальных данных
│   ├── ExpHeatOldData.m                   # Старые данные нагрева
│   ├── ExpHeatWith2Dim.m                  # Двумерный вход
│   ├── ExpHeatWithHiddenCore.m            # Скрытое ядро (K > 1, цепочка с контекстом)
│   ├── ExpWith2Dim3Vertices.m             # 3 вершины, 2D-вход, multi-stage
│   ├── ExpLinear.m, ExpLinearTwoX.m       # Линейные эксперименты
│   ├── ExpLinear2026.m                    # Линейный эксперимент 2026
│   ├── ExpSigmoid.m                       # Сигмоидные ядра
│   ├── ExpSimpleData.m                    # Простые данные
│   ├── ExpTunableVerification.m           # Верификация настраиваемых функций
│   ├── ExpVerificationTheorem.m           # Численная проверка теоремы о декомпозиции
│   ├── ExpStructuralSearchTest.m          # Тест жадного NAS
│   ├── ExpStructuralSearchTest3Vertices.m # Тест жадного NAS на 3 вершинах
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
| `GammaCtx` | `double` (default=1) | Вес контекста от предыдущего этапа (K > 1) |
| `FResult` | `double` | Текущее значение в вершине (вычисляется `Forward`-проходом) |

Активация применяется только к выходу ядровой функции:

$$L_v = \sigma(\gamma_v \cdot f_v(x_v))$$

где σ — функция активации, f_v — ядровая функция. L_v входит в систему линейных уравнений прямого прохода (см. ниже). Вклады входящих рёбер добавляются линейно, без активации.

При K > 1 вход ядровой функции расширяется контекстом: `f_v(AugmentInput(x_v, ctx_vec))`, где `ctx_vec` — вектор F-значений входящих соседей с предыдущего этапа, масштабированный на `GammaCtx`.

### Рёбра (`Edge`)

Каждое ребро `u → v` несёт два параметра:

| Параметр | Роль |
|---|---|
| **α** (Alfa) | Линейный коэффициент передачи: вклад `F_u` в `F_v` умножается на α |
| **β** (Beta) | Аддитивное смещение |

Генерация α — публичный метод `GenerateTopologyAwareAlpha(SafetyFactor)`. Распределяет α так, чтобы ∀v выполнялось **условие устойчивости**:

$$\sum_{e \in \text{In}(v)} \alpha_e \;<\; 1 + \sum_{e \in \text{Out}(v)} \alpha_e$$

### Прямой проход (`Forward`)

Для каждой вершины v состояние F_v определяется балансом между активированным выходом ядровой функции и агрегированным сигналом соседей:

$$F_v = \frac{\tilde{L}_v + G_{In}(v) - \sum_{e \in E_{Out}(v)} \beta_e}{D(v)}, \qquad D(v) = 1 + \sum_{e \in E_{Out}(v)} \alpha_e$$

где $\tilde{L}_v = \sigma_v(\gamma_v \cdot f_v(X[v]))$ — активированный выход ядра, $G_{In}(v) = \sum_{e(u \to v)} (\alpha_e \cdot F_u + \beta_e)$ — агрегированный сигнал от входящих соседей.

**Матричная форма:** **(D − A_in) · F = L̃ + B_in − B_out**

- D = diag(D(v)) — диагональная матрица знаменателей
- (A_in)_{ij} = α_{j→i} — матрица входящих α
- B_in, B_out — вектора сумм β по входящим/исходящим рёбрам
- L̃ — вектор активированных выходов ядер

Решение: **F = (D − A_in)⁻¹ · (L̃ + B_in − B_out)**

**Кеширование:** M⁻¹ = (D − A_in)⁻¹ и const = B_in − B_out не зависят от входных данных — вычисляются **один раз** при изменении параметров рёбер и переиспользуются между сэмплами. Ускорение ~10×. Результат `CalcCoreFunction` также кешируется в `Node` между вызовами с одинаковыми входными данными.

### Многоэтапный прямой проход (K > 1)

При `GraphShell.NumStages > 1` (задаётся через `TrainingOptions.ContextStages`) выполняется K итераций прямого прохода с контекстной пропагацией:

**Плоский случай (K = 1):**
$$\mathbf{F} = (\mathbf{D} - \mathbf{A}_{in})^{-1} \bigl( \widetilde{\mathbf{L}} + \mathbf{B}_{in} - \mathbf{B}_{out} \bigr), \qquad \tilde{L}_v = \sigma_v(\gamma_v \cdot f_v(X[v]))$$

**Этап 1 (k = 1):** совпадает с плоским проходом. Вычисляется F⁽¹⁾.

**Этапы k = 2…K:** для каждой вершины v вычисляется вектор контекста из состояний входящих соседей с предыдущего этапа:
$$\mathbf{c}_v^{(k)} = \gamma^{Ctx}_v \cdot \bigl( F_{u_1}^{(k-1)}, F_{u_2}^{(k-1)}, \dots, F_{u_m}^{(k-1)} \bigr)^T \in \mathbb{R}^m$$

где m = |In(v)|, $\gamma^{Ctx}_v$ — обучаемый вес контекста (свойство `Node.GammaCtx`). Вход ядровой функции расширяется:
$$\tilde{L}_v^{(k)} = \sigma_v\bigl( \gamma_v \cdot f_v( \mathcal{A}_v(X[v], \mathbf{c}_v^{(k)}) ) \bigr)$$

где $\mathcal{A}_v$ = `AugmentInput` — оператор дополнения входа. Затем решается та же линейная система:
$$\mathbf{F}^{(k)} = (\mathbf{D} - \mathbf{A}_{in})^{-1} \bigl( \widetilde{\mathbf{L}}^{(k)} + \mathbf{B}_{in} - \mathbf{B}_{out} \bigr)$$

Финальное состояние: $\mathbf{F} = \mathbf{F}^{(K)}$.

На каждом этапе сохраняются промежуточные значения (`F_vector`, `ctx_store`, `raw_store`) для последующего BPTT. Матрица (D − A_in)⁻¹ факторизуется однократно и переиспользуется на всех этапах.

### Обратный проход: функция потерь чёрной вершины

Для чёрной вершины полная невязка складывается из структурной и собственной составляющих:

$$J_b = \lambda_{struct} \cdot J_b^{struct} + \lambda_{self} \cdot J_b^{self}$$

**Собственная структурная невязка** измеряет рассогласование между фактическим состоянием вершины и «теневым» значением, построенным только по состояниям соседей (без собственного аппроксиматора):

$$\tilde{F}_b = \frac{G_{in}(b) - \sum_{e \in Out(b)} \beta_e}{\sum_{e \in Out(b)} \alpha_e}, \qquad J_b^{self} = F_b - \tilde{F}_b$$

При Σα_out = 0 теневое значение не определено — J_self = 0.

### BPTT: обратное распространение через этапы

При K > 1 градиент по $\gamma^{Ctx}_v$ получает дополнительные BPTT-слагаемые, отсутствующие в плоской модели:

$$\frac{\partial J}{\partial \gamma^{Ctx}_v} = \sum_{k=2}^{K} \frac{\partial J}{\partial F^{(K)}} \cdot \frac{\partial F^{(K)}}{\partial F_v^{(k)}} \cdot \frac{\partial F_v^{(k)}}{\partial \mathbf{c}_v^{(k)}} \cdot \frac{\partial \mathbf{c}_v^{(k)}}{\partial \gamma^{Ctx}_v}$$

Сомножители раскрываются:

$$\frac{\partial F_v^{(k)}}{\partial \mathbf{c}_v^{(k)}} = [(\mathbf{D} - \mathbf{A}_{in})^{-1}]_{vv} \cdot \gamma_v \cdot \sigma_v'(z_v^{(k)}) \cdot \frac{\partial f_v}{\partial \mathbf{c}_v^{(k)}}$$

$$\frac{\partial \mathbf{c}_v^{(k)}}{\partial \gamma^{Ctx}_v} = \bigl( F_{u_1}^{(k-1)}, \dots, F_{u_m}^{(k-1)} \bigr)^T, \qquad \frac{\partial \mathbf{c}_v^{(k)}}{\partial F_{u_j}^{(k-1)}} = \gamma^{Ctx}_v \cdot \mathbf{e}_j$$

Производная $\partial f_v / \partial \mathbf{c}_v^{(k)}$ вычисляется через `ICoreF.CalcContextDerivative` (аналитически или центральной конечной разностью).

Алгоритм BPTT в `GraphShell.BackpropContext(J_total)`:
1. Начальное состояние: `dF = J_total` (градиент финальной ошибки по F каждой вершины)
2. Для этапов s = K…2:
   - Для каждой вершины i: `dL_i = dF(i) × [(D − A_in)⁻¹]_{ii}` — пропагация через линейную систему
   - `d_raw_i = dL_i × σ'(raw_store[i])` — через активацию
   - `dCore_dctx` = `node.CalcContextDerivative(baseInput, ctx_store[i])` — через ядро
   - `dL_dctx_i = d_raw_i × γ_i × dCore_dctx` — градиент L по контексту
   - `dF_dctx_i = dL_dctx_i × [(D − A_in)⁻¹]_{ii}` — пропагация на F
   - Градиент GammaCtx: `∇γCtx_i += dF_dctx_i · ctx_store[i] / γCtx_i`
   - Пропагация на F предыдущего этапа: `dF_prev(src) += γCtx_i × dF_dctx_i(j)` для каждого j-го соседа
3. Возврат `∇γCtx` — вектор градиентов для всех вершин

При K = 1 сумма в формуле градиента пуста, и BPTT-поправки тождественно равны нулю — многоэтапная модель является строгим обобщением плоской.

### Устойчивость

Жёсткое ограничение после каждого ADAM-шага: для каждой вершины v, если Σα_in(v) ≥ 1+Σα_out(v), все входящие α масштабируются с коэффициентом `StabilityClampFactor × (1+Σα_out) / Σα_in`. По умолчанию StabilityClampFactor = 0.99.

---

## Векторный контекст и ContextProjector

### Интерфейс контекста в `ICoreF`

Базовый класс `ICoreF` предоставляет три метода для поддержки многоэтапного прохода:

| Метод | По умолчанию | Описание |
|---|---|---|
| `SupportsContext()` | `false` | Может ли солвер использовать контекст от предыдущего этапа |
| `AugmentInput(baseInput, ctx_vec)` | `baseInput` | Расширяет входной вектор контекстом |
| `CalcContextDerivative(baseInput, ctx_vec)` | конечная разность | Производная `CalcCoreFunction` по вектору контекста (вектор-строка) |

**Реализация в `Heating2DModel`:**
- `SupportsContext → true`
- `AugmentInput` добавляет `ctx_vec(1)` как T₀ (начальную температуру)
- `CalcContextDerivative` возвращает `[1, 0, …, 0]` (dTavg/dT₀ = 1)

### ContextProjector

Специальная ядровая функция без собственного решателя — используется как промежуточный узел в многоэтапных цепочках:

- `CalcCoreFunction` возвращает первый элемент входного вектора (или 0, если вход пуст)
- `AugmentInput` возвращает вектор контекста, игнорируя базовый вход
- `CalcContextDerivative` возвращает единичную матрицу (identity)

**Типовая топология:** `A(решатель) → Proj(контекст) → B(решатель) → …`

Proj-вершина пропускает через себя F-значения входящих соседей, позволяя следующему решателю получить контекст от предыдущего этапа цепочки.

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
| `ContextStages` | 1 | Количество этапов Forward (K). K=1 — классический режим |
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
| `Lambda_Gamma` | 0.01 | L2-регуляризация γ (и GammaCtx при K>1) |
| `Lambda_Agg` | 0 | Штраф агрегации ошибок |
| `Lambda_Self` | 0 | Собственная регуляризация |
| `Lambda_Struct` | 0 | Структурная регуляризация |
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
| `StabilityClampFactor` | 0.99 | Коэффициент жёсткого ограничения α после ADAM |
| `StructInitAlpha` | 0.5 | Начальное α для новых рёбер |
| `StructInitBeta` | 1 | Начальное β для новых рёбер |

### Алгоритм 1: Выход из плато (Plateau Escape)

При застревании (отсутствие улучшения `p_e` эпох подряд):
1. К параметрам применяется случайное смещение (`RpShiftPercent`% от текущих значений, включая `GammaCtx`)
2. LR сбрасывается до начального
3. Максимум `p_p` попыток, затем остановка

LR-шедулинг: **η_epoch = η_init / √epoch** — затухание по корню эпохи для стабилизации сходимости.

### Алгоритм 2: Структурный поиск (Greedy NAS)

Двухфазный поиск:
1. **Фаза 1 (белые вершины):** добавление рёбер только к белым вершинам
2. **Фаза 2 (чёрные вершины):** добавление рёбер к чёрным вершинам

Каждые `StructuralSearchInterval` эпох:
1. Генерируются кандидаты — отсутствующие рёбра
2. Каждый кандидат быстро настраивается (`StructuralSearchEpochs` эпох)
3. Лучший кандидат (по ошибке на тесте) добавляется в граф
4. Глобальный кеш (`globalEdgeCache`) исключает повторную проверку отклонённых рёбер
5. После конвергенции — `CleanupRedundantEdges()` удаляет рёбра с α ниже порога

### Визуализация обучения

Дашборд 3×3 в реальном времени:
- **Строка 1:** ошибки train/test (с маркерами структурных изменений), learning rate, матрица смежности
- **Строка 2:** время эпохи, разница ошибок (early stopping), таблица вершин (γ, γ_C при K>1)
- **Строка 3:** таблица рёбер (α, β), визуальный граф (чистый, без таблиц)

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
| `Forward(Data)` | Прямой проход (K=1 — классический, K>1 — многоэтапный), обновляет `FResult` |
| `BackpropContext(dF_final)` | BPTT: градиент `GammaCtx` через этапы (K>1) |
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
| `DrawGraph_New(titleStr, ax, hideEdgeLabels)` | Визуализация графа. 4-й параметр скрывает метки рёбер |
| `DrawNodeTable(ax)` | Таблица параметров вершин (γ, γ_C) на заданных осях |
| `DrawEdgeTable(ax)` | Таблица параметров рёбер (α, β) на заданных осях |
| `DrawParamTables(ax)` | Таблица вершин в углу графика (для standalone-режима) |
| `GenerateTopologyAwareAlpha(SafetyFactor)` | Генерация α с гарантией устойчивости (публичный) |
| `invalidateForwardCache()` | Сброс кеша M⁻¹ и BPTT-кеша |
| `addEdgeBetween(src, dst, α, β)` | Добавить направленное ребро |
| `removeEdgeBetween(src, dst)` | Удалить направленное ребро |
| `hasEdge(src, dst)` | Проверить существование ребра |
| `getPossibleEdges()` | Все пары (i,j) без ребра |
| `getExistingEdges()` | Все существующие рёбра |
| `getTotalEdgeCount()` | Общее количество рёбер |
| `getEdgeParams(src, dst)` | Параметры (α, β) ребра |
| `generateEdgeParams()` | Сгенерировать (α, β) теми же генераторами |
| `checkStability()` | Проверить условие устойчивости для всех вершин |
| `clone()` | Глубокая копия графа |

**Свойства:**
| Свойство | Описание |
|---|---|
| `NumStages` | Количество этапов Forward (K). По умолчанию 1 |
| `ListOfNodes` | Вектор всех узлов |

### `Node`

| Метод | Описание |
|---|---|
| `Node(ID, initVal, nodeType, nodeFunction, activationType)` | Конструктор |
| `addEdge(targetNode)` | Добавить исходящее ребро |
| `removeEdgeByTarget(targetNode)` | Удалить ребро |
| `getEdgeToTarget(targetNode)` | Получить ребро |
| `getOutEdges()` / `getOutEdgesMap()` | Исходящие рёбра |
| `getNeighbors()` | Соседи |
| `calcNodeFunc(inputData)` | γ·CoreFunction → activation |
| `calcRawCoreFunction(inputData)` | CoreFunction (с кешированием) |
| `computeLGammaDerivative(inputData)` | ∂L/∂γ |
| `getFResult()` / `setFResult(v)` | Текущее значение |
| `getNodeType()` | `White` / `Black` |
| `getNodeFunction()` | Ядровая функция |
| `getActivationType()` | Тип активации |
| `getActivationDerivative(raw)` | Производная активации по raw |

**Свойства:**
| Свойство | По умолчанию | Описание |
|---|---|---|
| `ID` | — | Номер вершины |
| `Gamma` | 1 | Вес выхода CoreFunction |
| `GammaCtx` | 1 | Вес контекста (multi-stage, K>1) |
| `ActivationType` | `"linear"` | Тип нелинейности |

### `Edge`

| Свойство | Описание |
|---|---|
| `Alfa` | Линейный коэффициент α |
| `Beta` | Аддитивное смещение β |
| `SourceNode` / `TargetNode` | Вершины |
| `ID` | Идентификатор |

### `ICoreF` (интерфейс ядровой функции)

| Метод | Описание |
|---|---|
| `CalcCoreFunction(InputParams)` | Вычислить ядровую функцию |
| `GetNumOfInputParams()` | Количество входных параметров |
| `SupportsContext()` | Поддерживает ли контекст (default: false) |
| `AugmentInput(baseInput, ctx_vec)` | Расширить вход контекстом (default: baseInput) |
| `CalcContextDerivative(baseInput, ctx_vec)` | ∂Core/∂ctx — вектор-строка (default: конечная разность) |

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
- **ExpHeatWithHiddenCore** — скрытое ядро, многоэтапный проход (K > 1), цепочка с контекстом
- **ExpWith2Dim3Vertices** — 3 вершины, 2D-вход, multi-stage с BPTT
- **ExpStructuralSearchTest / ExpStructuralSearchTest3Vertices** — тест жадного NAS
- **ExpVerificationTheorem** — численная проверка теоремы о декомпозиции C_{b→w}
- **ExpHeatWith2Dim** — двумерный вход (t, Tinf)
- **ExpLinear / ExpLinearTwoX / ExpLinear2026** — линейные ядра
- **ExpSigmoid** — сигмоидная активация
- **ExpTunableVerification** — настраиваемые ядровые функции (ITunableCoreF)

---

## Результаты

На задаче последовательного моделирования (T₀-цепочка, 3 вершины, K=3 этапа) модель BW Graph показала:

| Метрика | BW Graph (K=3) | Gradient Boosting |
|---|---|---|
| R² | **0.9877** | 0.8582 |
| MAE | **9.0** | 29.9 |

Многоэтапный проход с BPTT и векторным контекстом позволяет модели улавливать зависимость от начальных условий (T₀), передаваемую через этапы цепочки.

---

## Технические аспекты

**Реализовано:**
- Мультипликативные (γ) и аддитивные (β) зависимости между ядрами
- Линейные коэффициенты передачи (α) с гарантией устойчивости
- Нелинейные активации: `linear`, `sigmoid`, `relu`, `tanh`
- ADAM-оптимизатор с раздельным клиппингом (α, β, γ, GammaCtx) и автокалибровкой
- Многоэтапный прямой проход (K > 1) с векторным контекстом между этапами
- BPTT (обратное распространение через этапы) для обучения `GammaCtx`
- `ContextProjector` — ядровая функция-проектор для цепочек без собственного решателя
- Механизм выхода из плато (RpShift) с поддержкой `GammaCtx`
- Жадный структурный поиск (NAS) с двухфазной стратегией и глобальным кешем
- Кеширование прямого прохода (M⁻¹), ядровых функций и промежуточных BPTT-состояний
- Сохранение/загрузка модели через `LoadFromFile`
