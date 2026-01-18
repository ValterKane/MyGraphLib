% Функция численного решения задачи нестационарного теплообмена в 2D сечении
% с использованием неявной схемы (метод Кранка-Николсон)
classdef Heating2DModel < coreFunctions.ICoreF
    properties
        Lx;         % Длина по оси X [м]
        Ly;         % Длина по оси Y [м]
        alpha;      % Коэффициент температуропроводности [м^2/с]
        lambda;     % Теплопроводность материала [Вт/(м·K)]
        h;          % Коэффициент теплоотдачи [Вт/(м^2·K)]
        nx;         % Количество узлов по оси X
        ny;         % Количество узлов по оси Y
        T0;         % Начальная температура
        nt;         % Количество шагов по времени
    end

    methods
        function obj = Heating2DModel(h, nx, ny, lambda, alpha, Lx, Ly, T0, nt)
            % Конструктор класса
            obj.Lx = Lx;
            obj.Ly = Ly;
            obj.alpha = alpha;
            obj.lambda = lambda;
            obj.h = h;
            obj.nx = nx;
            obj.ny = ny;
            obj.T0 = T0;
            obj.nt = nt;
        end

        function result = CalcCoreFunction(obj, InputParams)
            arguments
                obj
                InputParams (2,1) double {mustBePositive}
            end

            % Извлечение параметров
            T_inf = InputParams(2,1);  % Температура окружающей среды
            time = InputParams(1,1);   % Время

            % Расчет шагов
            dx = obj.Lx/(obj.nx-1);
            dy = obj.Ly/(obj.ny-1);
            dt = time/obj.nt;

            % Инициализация температурного поля
            T = ones(obj.nx * obj.ny, 1) * obj.T0;

            % Построение матрицы системы и правой части для неявной схемы
            N = obj.nx * obj.ny;
            A = sparse(N, N);
            b = zeros(N, 1);

            % Коэффициенты для метода Кранка-Николсон (theta = 0.5)
            theta = 0.5;
            rx = obj.alpha * dt / dx^2;
            ry = obj.alpha * dt / dy^2;

            % Основной цикл по времени
            for k = 1:obj.nt
                % Обнуляем матрицу и правую часть
                A = sparse(N, N);
                b = zeros(N, 1);

                % Заполняем матрицу системы и правую часть
                for i = 1:obj.nx
                    for j = 1:obj.ny
                        idx = (j-1)*obj.nx + i;  % Линеаризованный индекс

                        if i > 1 && i < obj.nx && j > 1 && j < obj.ny
                            % Внутренние точки
                            A(idx, idx) = 1 + theta*(rx + ry);
                            A(idx, idx-1) = -theta*rx/2;
                            A(idx, idx+1) = -theta*rx/2;
                            A(idx, idx-obj.nx) = -theta*ry/2;
                            A(idx, idx+obj.nx) = -theta*ry/2;

                            % Правая часть
                            b(idx) = T(idx) + (1-theta)*rx*(T(idx-1) - 2*T(idx) + T(idx+1))/2 ...
                                + (1-theta)*ry*(T(idx-obj.nx) - 2*T(idx) + T(idx+obj.nx))/2;
                        elseif i == 1 && j > 1 && j < obj.ny
                            % Левая граница: -lambda*dT/dx = h*(T_inf - T)
                            A(idx, idx) = 1 + theta*rx + theta*ry/2 + theta*rx*dx*obj.h/obj.lambda;
                            A(idx, idx+1) = -theta*rx;
                            A(idx, idx-obj.nx) = -theta*ry/4;
                            A(idx, idx+obj.nx) = -theta*ry/4;

                            b(idx) = T(idx) + (1-theta)*rx*(T(idx+1) - T(idx) + dx*obj.h/obj.lambda*(T_inf - T(idx))) ...
                                + (1-theta)*ry*(T(idx-obj.nx) - 2*T(idx) + T(idx+obj.nx))/2 ...
                                + theta*rx*dx*obj.h/obj.lambda*T_inf;
                        elseif i == obj.nx && j > 1 && j < obj.ny
                            % Правая граница: lambda*dT/dx = h*(T_inf - T)
                            A(idx, idx) = 1 + theta*rx + theta*ry/2 + theta*rx*dx*obj.h/obj.lambda;
                            A(idx, idx-1) = -theta*rx;
                            A(idx, idx-obj.nx) = -theta*ry/4;
                            A(idx, idx+obj.nx) = -theta*ry/4;

                            b(idx) = T(idx) + (1-theta)*rx*(T(idx-1) - T(idx) + dx*obj.h/obj.lambda*(T_inf - T(idx))) ...
                                + (1-theta)*ry*(T(idx-obj.nx) - 2*T(idx) + T(idx+obj.nx))/2 ...
                                + theta*rx*dx*obj.h/obj.lambda*T_inf;
                        elseif j == obj.ny && i > 1 && i < obj.nx
                            % Верхняя граница: lambda*dT/dy = h*(T_inf - T)
                            A(idx, idx) = 1 + theta*rx/2 + theta*ry + theta*ry*dy*obj.h/obj.lambda;
                            A(idx, idx-1) = -theta*rx/4;
                            A(idx, idx+1) = -theta*rx/4;
                            A(idx, idx-obj.nx) = -theta*ry;

                            b(idx) = T(idx) + (1-theta)*rx*(T(idx-1) - 2*T(idx) + T(idx+1))/2 ...
                                + (1-theta)*ry*(T(idx-obj.nx) - T(idx) + dy*obj.h/obj.lambda*(T_inf - T(idx))) ...
                                + theta*ry*dy*obj.h/obj.lambda*T_inf;
                        elseif j == 1 && i > 1 && i < obj.nx
                            % Нижняя граница: адиабатическая (dT/dy = 0)
                            A(idx, idx) = 1 + theta*rx/2 + theta*ry;
                            A(idx, idx-1) = -theta*rx/4;
                            A(idx, idx+1) = -theta*rx/4;
                            A(idx, idx+obj.nx) = -theta*ry;

                            b(idx) = T(idx) + (1-theta)*rx*(T(idx-1) - 2*T(idx) + T(idx+1))/2 ...
                                + (1-theta)*ry*(T(idx+obj.nx) - T(idx));
                        elseif i == 1 && j == 1
                            % Левый нижний угол: адиабатическая снизу + граница слева
                            A(idx, idx) = 1 + theta*rx + theta*ry + theta*rx*dx*obj.h/obj.lambda;
                            A(idx, idx+1) = -theta*rx;
                            A(idx, idx+obj.nx) = -theta*ry;

                            b(idx) = T(idx) + (1-theta)*rx*(T(idx+1) - T(idx) + dx*obj.h/obj.lambda*(T_inf - T(idx))) ...
                                + (1-theta)*ry*(T(idx+obj.nx) - T(idx)) ...
                                + theta*rx*dx*obj.h/obj.lambda*T_inf;
                        elseif i == obj.nx && j == 1
                            % Правый нижний угол: адиабатическая снизу + граница справа
                            A(idx, idx) = 1 + theta*rx + theta*ry + theta*rx*dx*obj.h/obj.lambda;
                            A(idx, idx-1) = -theta*rx;
                            A(idx, idx+obj.nx) = -theta*ry;

                            b(idx) = T(idx) + (1-theta)*rx*(T(idx-1) - T(idx) + dx*obj.h/obj.lambda*(T_inf - T(idx))) ...
                                + (1-theta)*ry*(T(idx+obj.nx) - T(idx)) ...
                                + theta*rx*dx*obj.h/obj.lambda*T_inf;
                        elseif i == 1 && j == obj.ny
                            % Левый верхний угол: граница слева + сверху
                            A(idx, idx) = 1 + theta*rx + theta*ry + theta*rx*dx*obj.h/obj.lambda + theta*ry*dy*obj.h/obj.lambda;
                            A(idx, idx+1) = -theta*rx;
                            A(idx, idx-obj.nx) = -theta*ry;

                            b(idx) = T(idx) + (1-theta)*rx*(T(idx+1) - T(idx) + dx*obj.h/obj.lambda*(T_inf - T(idx))) ...
                                + (1-theta)*ry*(T(idx-obj.nx) - T(idx) + dy*obj.h/obj.lambda*(T_inf - T(idx))) ...
                                + theta*rx*dx*obj.h/obj.lambda*T_inf ...
                                + theta*ry*dy*obj.h/obj.lambda*T_inf;
                        elseif i == obj.nx && j == obj.ny
                            % Правый верхний угол: граница справа + сверху
                            A(idx, idx) = 1 + theta*rx + theta*ry + theta*rx*dx*obj.h/obj.lambda + theta*ry*dy*obj.h/obj.lambda;
                            A(idx, idx-1) = -theta*rx;
                            A(idx, idx-obj.nx) = -theta*ry;

                            b(idx) = T(idx) + (1-theta)*rx*(T(idx-1) - T(idx) + dx*obj.h/obj.lambda*(T_inf - T(idx))) ...
                                + (1-theta)*ry*(T(idx-obj.nx) - T(idx) + dy*obj.h/obj.lambda*(T_inf - T(idx))) ...
                                + theta*rx*dx*obj.h/obj.lambda*T_inf ...
                                + theta*ry*dy*obj.h/obj.lambda*T_inf;
                        end
                    end
                end

                % Решение системы линейных уравнений
                T = A \ b;

            end

            % Преобразование обратно в матрицу для вычисления средней температуры
            T_matrix = reshape(T, obj.nx, obj.ny);

            % Формирование результата - средняя температура
            result = mean(T_matrix, 'all');
        end

        function result = GetNumOfInputParams(obj)
            % Возвращает количество входных параметров
            result = 2; % time, T_inf, дополнительный параметр
        end

        function visualizeTemperature(obj, InputParams)
            % Визуализация температурного поля
            T_inf = InputParams(2,1);
            time = InputParams(1,1);

            % Повторяем расчет для визуализации
            dx = obj.Lx/(obj.nx-1);
            dy = obj.Ly/(obj.ny-1);
            dt = time/obj.nt;

            N = obj.nx * obj.ny;
            T = ones(N, 1) * obj.T0;

            % Коэффициенты для метода Кранка-Николсон
            theta = 0.5;
            rx = obj.alpha * dt / dx^2;
            ry = obj.alpha * dt / dy^2;

            for k = 1:obj.nt
                A = sparse(N, N);
                b = zeros(N, 1);

                for i = 1:obj.nx
                    for j = 1:obj.ny
                        idx = (j-1)*obj.nx + i;

                        if i > 1 && i < obj.nx && j > 1 && j < obj.ny
                            A(idx, idx) = 1 + theta*(rx + ry);
                            A(idx, idx-1) = -theta*rx/2;
                            A(idx, idx+1) = -theta*rx/2;
                            A(idx, idx-obj.nx) = -theta*ry/2;
                            A(idx, idx+obj.nx) = -theta*ry/2;

                            b(idx) = T(idx) + (1-theta)*rx*(T(idx-1) - 2*T(idx) + T(idx+1))/2 ...
                                + (1-theta)*ry*(T(idx-obj.nx) - 2*T(idx) + T(idx+obj.nx))/2;
                        elseif i == 1 && j > 1 && j < obj.ny
                            A(idx, idx) = 1 + theta*rx + theta*ry/2 + theta*rx*dx*obj.h/obj.lambda;
                            A(idx, idx+1) = -theta*rx;
                            A(idx, idx-obj.nx) = -theta*ry/4;
                            A(idx, idx+obj.nx) = -theta*ry/4;

                            b(idx) = T(idx) + (1-theta)*rx*(T(idx+1) - T(idx) + dx*obj.h/obj.lambda*(T_inf - T(idx))) ...
                                + (1-theta)*ry*(T(idx-obj.nx) - 2*T(idx) + T(idx+obj.nx))/2 ...
                                + theta*rx*dx*obj.h/obj.lambda*T_inf;
                        elseif i == obj.nx && j > 1 && j < obj.ny
                            A(idx, idx) = 1 + theta*rx + theta*ry/2 + theta*rx*dx*obj.h/obj.lambda;
                            A(idx, idx-1) = -theta*rx;
                            A(idx, idx-obj.nx) = -theta*ry/4;
                            A(idx, idx+obj.nx) = -theta*ry/4;

                            b(idx) = T(idx) + (1-theta)*rx*(T(idx-1) - T(idx) + dx*obj.h/obj.lambda*(T_inf - T(idx))) ...
                                + (1-theta)*ry*(T(idx-obj.nx) - 2*T(idx) + T(idx+obj.nx))/2 ...
                                + theta*rx*dx*obj.h/obj.lambda*T_inf;
                        elseif j == obj.ny && i > 1 && i < obj.nx
                            A(idx, idx) = 1 + theta*rx/2 + theta*ry + theta*ry*dy*obj.h/obj.lambda;
                            A(idx, idx-1) = -theta*rx/4;
                            A(idx, idx+1) = -theta*rx/4;
                            A(idx, idx-obj.nx) = -theta*ry;

                            b(idx) = T(idx) + (1-theta)*rx*(T(idx-1) - 2*T(idx) + T(idx+1))/2 ...
                                + (1-theta)*ry*(T(idx-obj.nx) - T(idx) + dy*obj.h/obj.lambda*(T_inf - T(idx))) ...
                                + theta*ry*dy*obj.h/obj.lambda*T_inf;
                        elseif j == 1 && i > 1 && i < obj.nx
                            A(idx, idx) = 1 + theta*rx/2 + theta*ry;
                            A(idx, idx-1) = -theta*rx/4;
                            A(idx, idx+1) = -theta*rx/4;
                            A(idx, idx+obj.nx) = -theta*ry;

                            b(idx) = T(idx) + (1-theta)*rx*(T(idx-1) - 2*T(idx) + T(idx+1))/2 ...
                                + (1-theta)*ry*(T(idx+obj.nx) - T(idx));
                        elseif i == 1 && j == 1
                            A(idx, idx) = 1 + theta*rx + theta*ry + theta*rx*dx*obj.h/obj.lambda;
                            A(idx, idx+1) = -theta*rx;
                            A(idx, idx+obj.nx) = -theta*ry;

                            b(idx) = T(idx) + (1-theta)*rx*(T(idx+1) - T(idx) + dx*obj.h/obj.lambda*(T_inf - T(idx))) ...
                                + (1-theta)*ry*(T(idx+obj.nx) - T(idx)) ...
                                + theta*rx*dx*obj.h/obj.lambda*T_inf;
                        elseif i == obj.nx && j == 1
                            A(idx, idx) = 1 + theta*rx + theta*ry + theta*rx*dx*obj.h/obj.lambda;
                            A(idx, idx-1) = -theta*rx;
                            A(idx, idx+obj.nx) = -theta*ry;

                            b(idx) = T(idx) + (1-theta)*rx*(T(idx-1) - T(idx) + dx*obj.h/obj.lambda*(T_inf - T(idx))) ...
                                + (1-theta)*ry*(T(idx+obj.nx) - T(idx)) ...
                                + theta*rx*dx*obj.h/obj.lambda*T_inf;
                        elseif i == 1 && j == obj.ny
                            A(idx, idx) = 1 + theta*rx + theta*ry + theta*rx*dx*obj.h/obj.lambda + theta*ry*dy*obj.h/obj.lambda;
                            A(idx, idx+1) = -theta*rx;
                            A(idx, idx-obj.nx) = -theta*ry;

                            b(idx) = T(idx) + (1-theta)*rx*(T(idx+1) - T(idx) + dx*obj.h/obj.lambda*(T_inf - T(idx))) ...
                                + (1-theta)*ry*(T(idx-obj.nx) - T(idx) + dy*obj.h/obj.lambda*(T_inf - T(idx))) ...
                                + theta*rx*dx*obj.h/obj.lambda*T_inf ...
                                + theta*ry*dy*obj.h/obj.lambda*T_inf;
                        elseif i == obj.nx && j == obj.ny
                            A(idx, idx) = 1 + theta*rx + theta*ry + theta*rx*dx*obj.h/obj.lambda + theta*ry*dy*obj.h/obj.lambda;
                            A(idx, idx-1) = -theta*rx;
                            A(idx, idx-obj.nx) = -theta*ry;

                            b(idx) = T(idx) + (1-theta)*rx*(T(idx-1) - T(idx) + dx*obj.h/obj.lambda*(T_inf - T(idx))) ...
                                + (1-theta)*ry*(T(idx-obj.nx) - T(idx) + dy*obj.h/obj.lambda*(T_inf - T(idx))) ...
                                + theta*rx*dx*obj.h/obj.lambda*T_inf ...
                                + theta*ry*dy*obj.h/obj.lambda*T_inf;
                        end
                    end
                end

                T = A \ b;
            end

            % Преобразование для визуализации
            T_matrix = reshape(T, obj.nx, obj.ny);

            % Визуализация
            figure;
            imagesc([0 obj.Lx], [0 obj.Ly], T_matrix');
            colorbar;
            title(sprintf('Температурное поле (неявная схема) при t = %.2f с', time));
            xlabel('x [м]');
            ylabel('y [м]');
            axis equal tight;
            colormap('jet');
        end

        function data = save(obj)
            % Сериализация объекта
            data.Lx = obj.Lx;
            data.Ly = obj.Ly;
            data.alpha = obj.alpha;
            data.lambda = obj.lambda;
            data.h = obj.h;
            data.nx = obj.nx;
            data.ny = obj.ny;
            data.T0 = obj.T0;
            data.nt = obj.nt;
        end

        function load(obj, data)
            % Десериализация объекта
            obj.Lx = data.Lx;
            obj.Ly = data.Ly;
            obj.alpha = data.alpha;
            obj.lambda = data.lambda;
            obj.h = data.h;
            obj.nx = data.nx;
            obj.ny = data.ny;
            obj.T0 = data.T0;
            obj.nt = data.nt;
        end
    end

    methods (Static)
        function obj = createFromData(data)
            % Создание объекта из данных
            obj = PlateHeating2DModel(...
                data.h, data.nx, data.ny, data.lambda, data.alpha, ...
                data.Lx, data.Ly, data.T0, data.nt);
        end
    end
end