# Экспериментальная библиотека для модели черно-белого графа (v.0.2)
Структура проекта
```
    MyGraphLib
    
    ├───+BWGraph                            # Пространство имен модели черно-белого графа
    │   │   Edge.m                          # Класс, описывающие рёбра графа
    │   │   GraphShell.m                    # Класс-оболочка графа
    │   │   Node.m                          # Класс, описывающий вершины графа
    │   │   NodeColor.m                     # Перечисление для определения типа вершины (черная / белая)
    │   │
    │   ├───+CustomMatrix                   # Пространство имен для описания матрицы из векторов произвольной длины
    │   │       BWMatrix.m                  # Класс, описывающий матрицу и базовые операции
    │   │       BWRow.m                     # Класс, описывающий вектор-строку произвольной длины
    │   │
    │   ├───+RandomGenerator                # Пространство имен для генераторов первичной инициализации параметров рёбер
    │   │       AlphaGenerator.m            # Пример класса для инициализации параметров альфа
    │   │       BetaGenerator.m             # пример класса для инициализации параметров бета
    │   │       IRandomGen.m                # Интерфейс, имплементация которого обязательна для всех будущих генераторов
    │   │
    │   └───+Trainer                        # Пространство имен, содержащие оболочку, контролирующую настройку модели
    │           Trainer.m                   # Класс, контролирующий настройку модели
    │
    ├───+coreFunctions                      # Пространство имен, содержащие пример ядровых функций и интерфейс для них
    │       HeatTransferBC.m                # Пример ядровой функции на основе уравнения теплового баланса
    │       ICoreF.m                        # Интерфейс ядровых функций, имплементация которого обязательна для всех будущих ядровых функций
    │       LinearFunction.m                # Пример простой линейной функции ядра
    │       LinearRegression.m              # Пример регрессионной функции ядра
    │       SigmoidFunction.m               # Пример сигмоидальной функции ядра
    └───    SimpleAddingCoreFunction.m      # Пример суммирующей функции ядра
  
```

## Examples
```matlab
    rng(42); % Фиксируем seed для воспроизводимости
    % Инициализация
    A = [0 1 1 1; 1 0 1 1; 1 1 0 1; 1 1 1 0];
    num_vertex = size(A, 1);
    
    % Создание модели
    HeatBC = coreFunctions.HeatTransferBC(320.0, 0.72);
    model_shell = statModel.ModelShell(A, 30, HeatBC);
    op = statModel.OptimizationMachine(model_shell);
    
    % Генерация данных с учетом индивидуальных характеристик вершин
    num_samples = 100;
    XData = zeros(HeatBC.GetNumOfInputParams, num_vertex, num_samples);
    YData = zeros(1, num_vertex, num_samples);
    
    for i = 1:num_samples
        target = 1100 + randi([-50,50],1,4);
        infTemp = 1050 + randi([-30,30],1,4);
        surfTemp = 1200 + randi([-40,40],1,4);
        
        XData(:,:,i) = [target; infTemp; surfTemp];
        YData(:,:,i) = target;
    end
    
    indices = randperm(num_samples);
    split_point = round(0.8 * num_samples);
    train_indices = indices(1:split_point);
    test_indices = indices(split_point+1:end);
    
    % Определим обучающую и тестовую выборку
    XData_train = XData(:, :, train_indices);
    YData_train = YData(:, :, train_indices);
    XData_test = XData(:, :, test_indices);
    YData_test = YData(:, :, test_indices);
    
    % Индивидуальные параметры для вершин
    vertex_regul_weight = [1.2, 1.1, 1.3, 1.1]; % Коэффициенты 
    vertex_weights = [1.0, 1.0, 1.0, 1.0]; % Весовые коэффициенты вершин
    
    % Вызываем цикл настройки
    op.Train(XData_train, YData_train, XData_test, YData_test, 32, 0.001, 0.9, 0.99, 1e-8, vertex_weights, ...
        vertex_regul_weight, 5000);
    
    % Тестирование
    model = op.GetModelShell();
    test_err = op.test_errors;
    
    test_output = YData(1,:,size(YData,3));
    test_input = XData(:,:,size(XData,3));
    model.Forward(test_input);
    total_result = model.GetResult();
    
    fprintf('Результат по лучшим параметрам:\nA1 = %.1f\nA2 = %.1f\nA3 = %.1f\nA4 = %.1f\n', total_result);
    fprintf('Ожидаемые результаты:\nA1 = %.1f\nA2 = %.1f\nA3 = %.1f\nA4 = %.1f\n', test_output);
    fprintf('Ошибки:\nA1 = %.1f\nA2 = %.1f\nA3 = %.1f\nA4 = %.1f\n', total_result - test_output);
    fprintf('Средняя ошибка:\nMAE = %.1f\n', mae(total_result,test_output));
    
    plot(test_err);
```
