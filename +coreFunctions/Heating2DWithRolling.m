% Функция численного решения задачи нестационарного сложного теплообмена в 2D сечении
% с использованием неявной схемы (метод Кранка-Николсон) и линеаризацией излучения
% + Интегрированная модель нагрева от пластической деформации в клетях
classdef Heating2DWithRolling < coreFunctions.ICoreF
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

        % Параметры для модели нагрева в клетях
        Cp_steel;           % Удельная теплоемкость стали [Дж/(кг·K)]
        motor_U;            % Напряжение питания двигателей [В]
        motor_efficiency;   % КПД двигателя
        mechanical_efficiency; % КПД преобразования механической энергии в тепло деформации
        enable_rolling_heating; % Флаг включения/выключения нагрева в клетях
    end
    methods
        function obj = Heating2DWithRolling(h_conv, epsilon, nx, ny, lambda, alpha, Lx, Ly, T0, nt)
            % Конструктор класса
            % Добавлены опциональные параметры для нагрева в клетях
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
            obj.Cp_steel = 600;
            obj.motor_efficiency = 0.9;
            obj.motor_U = 380;
            obj.mechanical_efficiency = 0.7;
        end

        function result = CalcCoreFunction(obj, InputParams)
            arguments
                obj
                InputParams (5,1) double {mustBePositive}
            end
            time = InputParams(1,1);   % Время
            T_inf = InputParams(2,1);  % Температура газов (окружающей среды)
            currents = InputParams(3,1);
            rpm = InputParams(4,1);
            weight = InputParams(5,1);
            time_stand = 10;

            % Расчет нагрева в печи
            T_after_furnace = obj.calcFurnaceHeating(time, T_inf);

            % Расчет нагрева в клетях
            T_final = obj.calcRollingHeating(T_after_furnace, currents, rpm, time_stand, weight);
            result = T_final;
        end

        function T_final = calcFurnaceHeating(obj, time, T_inf)
            % Расчет температурного поля при нагреве в печи
            % (оригинальная логика из CalcCoreFunction)

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

            % Возвращаем среднюю температуру после печного нагрева
            T_final = mean(T_matrix, 'all');
        end

        function T_after_rolling = calcRollingHeating(obj, T_after_furnace, currents, rpm, time_in_stand, weight)
            % Расчет нагрева заготовки в клетях прокатного стана
            %
            % Входные параметры:
            %   T_after_furnace - температура после печи (средняя по сечению) [K]
            %   currents - вектор токов двигателей клетей [А]
            %   rpm - вектор скоростей вращения валков [об/мин]
            %   time_in_stand - время обработки в каждой клети [с]
            %
            % Выход:
            %   T_after_rolling - температура после всех клетей [K]

            if ~obj.enable_rolling_heating
                warning('Нагрев в клетях отключен. Возвращается температура после печи.');
                T_after_rolling = T_after_furnace;
                return;
            end

            % Проверка размерностей входных данных
            N_cells = length(currents);
            if length(rpm) ~= N_cells || length(time_in_stand) ~= N_cells
                error('Векторы currents, rpm и time_in_stand должны иметь одинаковую длину');
            end

            T_current = T_after_furnace;
            rpm_to_rads = 2*pi/60;

            % Цикл по клетям
            for i = 1:N_cells
                I = currents(i);
                omega = rpm(i) * rpm_to_rads;
                dt = time_in_stand(i);

                % Электрическая мощность
                P_elec = I * obj.motor_U;

                % Механическая мощность на валу
                P_mech = P_elec * obj.motor_efficiency;

                % Работа за время dt
                W_mech = P_mech * dt;

                % Тепловая энергия, переданная заготовке
                Q_heat = W_mech * obj.mechanical_efficiency;

                % Повышение температуры
                dT = Q_heat / (weight * obj.Cp_steel);

                % Обновление температуры
                T_current = T_current + dT;
            end

            T_after_rolling = T_current;

            % Защита от нефизичных значений
            if T_after_rolling > 2000
                warning(['Расчетная температура после клетей (%.1f K) превышает температуру ' ...
                    'плавления стали. Проверьте входные данные.'], T_after_rolling);
            end
        end

        function result = GetNumOfInputParams(obj)
            result = 5; 
        end

        function visualizeTemperature(obj, InputParams)
            % Визуализация температурного поля
            if length(InputParams) >= 2
                time = InputParams(1);
                T_inf = InputParams(2);

                % Расчет только печного нагрева для визуализации
                T = obj.calcFurnaceHeating(time, T_inf);
                fprintf('Средняя температура после печи: %.1f K (%.1f °C)\n', T, T - 273.15);

                if length(InputParams) >= 5 && obj.enable_rolling_heating
                    N_cells = (length(InputParams) - 2) / 3;
                    currents = InputParams(3:2+N_cells);
                    rpm = InputParams(3+N_cells:2+2*N_cells);
                    time_stand = InputParams(3+2*N_cells:end);

                    T_final = obj.calcRollingHeating(T, currents, rpm, time_stand);
                    fprintf('Средняя температура после клетей: %.1f K (%.1f °C)\n', ...
                        T_final, T_final - 273.15);
                    fprintf('Суммарный нагрев в клетях: %.1f K\n', T_final - T);
                end
            else
                warning('Недостаточно параметров для визуализации');
            end
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

            % Параметры нагрева в клетях
            data.weight = obj.weight;
            data.Cp_steel = obj.Cp_steel;
            data.motor_U = obj.motor_U;
            data.motor_efficiency = obj.motor_efficiency;
            data.mechanical_efficiency = obj.mechanical_efficiency;
            data.enable_rolling_heating = obj.enable_rolling_heating;
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

            % Параметры нагрева в клетях
            if isfield(data, 'weight')
                obj.weight = data.weight;
                obj.Cp_steel = data.Cp_steel;
                obj.motor_U = data.motor_U;
                obj.motor_efficiency = data.motor_efficiency;
                obj.mechanical_efficiency = data.mechanical_efficiency;
                obj.enable_rolling_heating = data.enable_rolling_heating;
            end
        end
    end
    methods (Static)
        function obj = createFromData(data)
            % Создание объекта из данных
            if isfield(data, 'weight')
                obj = Heating2DWithNonLinear(...
                    data.h_conv, data.epsilon, data.nx, data.ny, data.lambda, data.alpha, ...
                    data.Lx, data.Ly, data.T0, data.nt, ...
                    'weight', data.weight, 'Cp_steel', data.Cp_steel, ...
                    'motor_U', data.motor_U, 'motor_eff', data.motor_efficiency, ...
                    'mech_eff', data.mechanical_efficiency, ...
                    'enable_rolling', data.enable_rolling_heating);
            else
                obj = Heating2DWithNonLinear(...
                    data.h_conv, data.epsilon, data.nx, data.ny, data.lambda, data.alpha, ...
                    data.Lx, data.Ly, data.T0, data.nt);
            end
        end
    end
end