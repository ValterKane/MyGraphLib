% Функция численного решения задачи нестационарного сложного теплообмена в 2D сечении
% с использованием неявной схемы (метод Кранка-Николсон) и линеаризацией излучения
classdef Heating2DWithNonLinear < coreFunctions.ICoreF
    properties
        Lx;         % Длина по оси X [м]
        Ly;         % Длина по оси Y [м]
        alpha;      % Коэффициент температуропроводности [м^2/с]
        lambda;     % Теплопроводность материала [Вт/(м·K)]
        h_conv;     % Конвективный коэффициент теплоотдачи [Вт/(м^2·K)] (малая величина, ~10)
        epsilon;    % Приведенная степень черноты
        sigma;      % Постоянная Стефана-Больцмана [Вт/(м^2·K^4)]
        nx;         % Количество узлов по оси X
        ny;         % Количество узлов по оси Y
        T0;         % Начальная температура [K] (Важно: ВСЕ температуры теперь в Кельвинах!)
        nt;         % Количество шагов по времени
        max_iter;   % Максимальное число итераций для линеаризации излучения
        tol;        % Допуск для сходимости итераций
    end

    methods
        function obj = Heating2DWithNonLinear(h_conv, epsilon, nx, ny, lambda, alpha, Lx, Ly, T0, nt)
            % Конструктор класса
            obj.Lx = Lx;
            obj.Ly = Ly;
            obj.alpha = alpha;
            obj.lambda = lambda;
            obj.h_conv = h_conv;       % Обычно 5-15 Вт/м2К
            obj.epsilon = epsilon;
            obj.sigma = 5.67e-8;       % Вт/(м2*K4)
            obj.nx = nx;
            obj.ny = ny;
            obj.T0 = T0;               % Должна быть в Кельвинах (например, 293.15)
            obj.nt = nt;
            obj.max_iter = 15;         % Опыт показывает, что 10-15 итераций достаточно
            obj.tol = 1e-4;            % Относительное изменение температуры
        end

        function result = CalcCoreFunction(obj, InputParams)
            arguments
                obj
                InputParams (2,1) double {mustBePositive}
            end

            % Извлечение параметров
            % ВАЖНО: T_inf теперь передается в КЕЛЬВИНАХ
            T_inf = InputParams(2,1);  % Температура газов (окружающей среды)
            time = InputParams(1,1);   % Время

            % Расчет шагов
            dx = obj.Lx/(obj.nx-1);
            dy = obj.Ly/(obj.ny-1);
            dt = time/obj.nt;

            % Инициализация температурного поля
            T = ones(obj.nx * obj.ny, 1) * obj.T0;

            % Построение матрицы системы и правой части для неявной схемы
            N = obj.nx * obj.ny;

            % Коэффициенты для метода Кранка-Николсон (theta = 0.5)
            theta = 0.5;
            rx = obj.alpha * dt / dx^2;
            ry = obj.alpha * dt / dy^2;

            % Основной цикл по времени
            for k = 1:obj.nt
                % Копия температуры для итерационного процесса (метод Пикара)
                T_old_iter = T;
                
                % Итерационный цикл для разрешения нелинейности излучения
                for iter = 1:obj.max_iter
                    A = sparse(N, N);
                    b = zeros(N, 1);

                    % Заполняем матрицу системы и правую часть
                    for i = 1:obj.nx
                        for j = 1:obj.ny
                            idx = (j-1)*obj.nx + i;  % Линеаризованный индекс

                            % Расчет коэффициента лучистой теплоотдачи для текущего узла
                            % Используем температуру с предыдущей итерации (T_old_iter)
                            T_current = T_old_iter(idx);
                            if T_inf ~= T_current
                                % alpha_rad = epsilon * sigma * (T_inf^4 - T_current^4) / (T_inf - T_current)
                                alpha_rad = obj.epsilon * obj.sigma * (T_inf^2 + T_current^2) * (T_inf + T_current);
                            else
                                alpha_rad = 0;
                            end
                            
                            % Суммарный коэффициент теплоотдачи
                            h_sum = obj.h_conv + alpha_rad;

                            if i > 1 && i < obj.nx && j > 1 && j < obj.ny
                                % Внутренние точки
                                A(idx, idx) = 1 + theta*(rx + ry);
                                A(idx, idx-1) = -theta*rx/2;
                                A(idx, idx+1) = -theta*rx/2;
                                A(idx, idx-obj.nx) = -theta*ry/2;
                                A(idx, idx+obj.nx) = -theta*ry/2;

                                b(idx) = T(idx) + (1-theta)*rx*(T(idx-1) - 2*T(idx) + T(idx+1))/2 ...
                                    + (1-theta)*ry*(T(idx-obj.nx) - 2*T(idx) + T(idx+obj.nx))/2;
                                
                            elseif i == 1 && j > 1 && j < obj.ny
                                % Левая граница: -lambda*dT/dx = h_sum*(T_inf - T)
                                % Упрощенное приведение к безразмерному виду для матрицы
                                Bi_sum = h_sum * dx / obj.lambda;
                                
                                A(idx, idx) = 1 + theta*rx + theta*ry/2 + theta*rx*Bi_sum;
                                A(idx, idx+1) = -theta*rx;
                                A(idx, idx-obj.nx) = -theta*ry/4;
                                A(idx, idx+obj.nx) = -theta*ry/4;

                                b(idx) = T(idx) + (1-theta)*rx*(T(idx+1) - T(idx) + Bi_sum*(T_inf - T(idx))) ...
                                    + (1-theta)*ry*(T(idx-obj.nx) - 2*T(idx) + T(idx+obj.nx))/2 ...
                                    + theta*rx*Bi_sum*T_inf;
                                
                            elseif i == obj.nx && j > 1 && j < obj.ny
                                % Правая граница: lambda*dT/dx = h_sum*(T_inf - T)
                                Bi_sum = h_sum * dx / obj.lambda;
                                
                                A(idx, idx) = 1 + theta*rx + theta*ry/2 + theta*rx*Bi_sum;
                                A(idx, idx-1) = -theta*rx;
                                A(idx, idx-obj.nx) = -theta*ry/4;
                                A(idx, idx+obj.nx) = -theta*ry/4;

                                b(idx) = T(idx) + (1-theta)*rx*(T(idx-1) - T(idx) + Bi_sum*(T_inf - T(idx))) ...
                                    + (1-theta)*ry*(T(idx-obj.nx) - 2*T(idx) + T(idx+obj.nx))/2 ...
                                    + theta*rx*Bi_sum*T_inf;
                                
                            elseif j == obj.ny && i > 1 && i < obj.nx
                                % Верхняя граница: lambda*dT/dy = h_sum*(T_inf - T)
                                Bi_sum = h_sum * dy / obj.lambda;
                                
                                A(idx, idx) = 1 + theta*rx/2 + theta*ry + theta*ry*Bi_sum;
                                A(idx, idx-1) = -theta*rx/4;
                                A(idx, idx+1) = -theta*rx/4;
                                A(idx, idx-obj.nx) = -theta*ry;

                                b(idx) = T(idx) + (1-theta)*rx*(T(idx-1) - 2*T(idx) + T(idx+1))/2 ...
                                    + (1-theta)*ry*(T(idx-obj.nx) - T(idx) + Bi_sum*(T_inf - T(idx))) ...
                                    + theta*ry*Bi_sum*T_inf;
                                
                            elseif j == 1 && i > 1 && i < obj.nx
                                % Нижняя граница: адиабатическая (dT/dy = 0)
                                A(idx, idx) = 1 + theta*rx/2 + theta*ry;
                                A(idx, idx-1) = -theta*rx/4;
                                A(idx, idx+1) = -theta*rx/4;
                                A(idx, idx+obj.nx) = -theta*ry;

                                b(idx) = T(idx) + (1-theta)*rx*(T(idx-1) - 2*T(idx) + T(idx+1))/2 ...
                                    + (1-theta)*ry*(T(idx+obj.nx) - T(idx));
                                
                            elseif i == 1 && j == 1
                                % Левый нижний угол
                                Bi_sum_x = h_sum * dx / obj.lambda;
                                A(idx, idx) = 1 + theta*rx + theta*ry + theta*rx*Bi_sum_x;
                                A(idx, idx+1) = -theta*rx;
                                A(idx, idx+obj.nx) = -theta*ry;

                                b(idx) = T(idx) + (1-theta)*rx*(T(idx+1) - T(idx) + Bi_sum_x*(T_inf - T(idx))) ...
                                    + (1-theta)*ry*(T(idx+obj.nx) - T(idx)) ...
                                    + theta*rx*Bi_sum_x*T_inf;
                                
                            elseif i == obj.nx && j == 1
                                % Правый нижний угол
                                Bi_sum_x = h_sum * dx / obj.lambda;
                                A(idx, idx) = 1 + theta*rx + theta*ry + theta*rx*Bi_sum_x;
                                A(idx, idx-1) = -theta*rx;
                                A(idx, idx+obj.nx) = -theta*ry;

                                b(idx) = T(idx) + (1-theta)*rx*(T(idx-1) - T(idx) + Bi_sum_x*(T_inf - T(idx))) ...
                                    + (1-theta)*ry*(T(idx+obj.nx) - T(idx)) ...
                                    + theta*rx*Bi_sum_x*T_inf;
                                
                            elseif i == 1 && j == obj.ny
                                % Левый верхний угол
                                Bi_sum_x = h_sum * dx / obj.lambda;
                                Bi_sum_y = h_sum * dy / obj.lambda;
                                
                                A(idx, idx) = 1 + theta*rx + theta*ry + theta*rx*Bi_sum_x + theta*ry*Bi_sum_y;
                                A(idx, idx+1) = -theta*rx;
                                A(idx, idx-obj.nx) = -theta*ry;

                                b(idx) = T(idx) + (1-theta)*rx*(T(idx+1) - T(idx) + Bi_sum_x*(T_inf - T(idx))) ...
                                    + (1-theta)*ry*(T(idx-obj.nx) - T(idx) + Bi_sum_y*(T_inf - T(idx))) ...
                                    + theta*rx*Bi_sum_x*T_inf ...
                                    + theta*ry*Bi_sum_y*T_inf;
                                
                            elseif i == obj.nx && j == obj.ny
                                % Правый верхний угол
                                Bi_sum_x = h_sum * dx / obj.lambda;
                                Bi_sum_y = h_sum * dy / obj.lambda;
                                
                                A(idx, idx) = 1 + theta*rx + theta*ry + theta*rx*Bi_sum_x + theta*ry*Bi_sum_y;
                                A(idx, idx-1) = -theta*rx;
                                A(idx, idx-obj.nx) = -theta*ry;

                                b(idx) = T(idx) + (1-theta)*rx*(T(idx-1) - T(idx) + Bi_sum_x*(T_inf - T(idx))) ...
                                    + (1-theta)*ry*(T(idx-obj.nx) - T(idx) + Bi_sum_y*(T_inf - T(idx))) ...
                                    + theta*rx*Bi_sum_x*T_inf ...
                                    + theta*ry*Bi_sum_y*T_inf;
                            end
                        end
                    end

                    % Решение системы линейных уравнений на текущей итерации
                    T_new = A \ b;
                    
                    % Проверка сходимости
                    diff = norm(T_new - T_old_iter) / norm(T_old_iter);
                    
                    % Обновление температуры для следующей итерации
                    T_old_iter = T_new;
                    
                    if diff < obj.tol
                        break;
                    end
                end
                
                % Сохраняем решение временного слоя
                T = T_old_iter;
                
                % Некоторая защита от физически невозможных значений
                if any(T < 0)
                    warning('Температура упала ниже абсолютного нуля. Проверьте входные данные.');
                end
            end

            % Преобразование обратно в матрицу для вычисления средней температуры
            T_matrix = reshape(T, obj.nx, obj.ny);

            % Формирование результата - средняя температура (возвращается в Кельвинах)
            result = mean(T_matrix, 'all');
        end

        function result = GetNumOfInputParams(obj)
            % Возвращает количество входных параметров
            result = 2; % time, T_inf
        end

        function visualizeTemperature(obj, InputParams)
            % Визуализация температурного поля (дублирует логику CalcCoreFunction)
            % Код опущен для краткости, он аналогичен CalcCoreFunction, 
            % но в конце строит график. Можно сделать refactoring.
            warning('Метод визуализации требует рефакторинга для нелинейной модели.');
        end

        function data = save(obj)
            % Сериализация объекта
            data.Lx = obj.Lx;
            data.Ly = obj.Ly;
            data.alpha = obj.alpha;
            data.lambda = obj.lambda;
            data.h_conv = obj.h_conv;
            data.epsilon = obj.epsilon;
            data.sigma = obj.sigma;
            data.nx = obj.nx;
            data.ny = obj.ny;
            data.T0 = obj.T0;
            data.nt = obj.nt;
            data.max_iter = obj.max_iter;
            data.tol = obj.tol;
        end

        function load(obj, data)
            % Десериализация объекта
            obj.Lx = data.Lx;
            obj.Ly = data.Ly;
            obj.alpha = data.alpha;
            obj.lambda = data.lambda;
            obj.h_conv = data.h_conv;
            obj.epsilon = data.epsilon;
            obj.sigma = data.sigma;
            obj.nx = data.nx;
            obj.ny = data.ny;
            obj.T0 = data.T0;
            obj.nt = data.nt;
            obj.max_iter = data.max_iter;
            obj.tol = data.tol;
        end
    end

    methods (Static)
        function obj = createFromData(data)
            % Создание объекта из данных
            obj = Heating2DModel(...
                data.h_conv, data.epsilon, data.nx, data.ny, data.lambda, data.alpha, ...
                data.Lx, data.Ly, data.T0, data.nt);
        end
    end
end