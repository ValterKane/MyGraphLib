% Настраиваемая версия Heating2DModel с обучаемыми параметрами
% h     — коэффициент теплоотдачи [Вт/(м^2·K)]
% alpha — коэффициент температуропроводности [м^2/с]
classdef Heating2DTunableModel < coreFunctions.ITunableCoreF
    properties
        Lx;             % Длина по оси X [м]
        Ly;             % Длина по оси Y [м]
        alpha;          % Коэффициент температуропроводности [м^2/с] (настраиваемый)
        lambda;         % Теплопроводность материала [Вт/(м·K)]
        h;              % Коэффициент теплоотдачи [Вт/(м^2·K)] (настраиваемый)
        nx;             % Количество узлов по оси X
        ny;             % Количество узлов по оси Y
        T0;             % Начальная температура
        nt;             % Количество шагов по времени
        learningRate;   % Скорость обучения для настройки параметров
    end

    properties (Hidden)
        m_h = 0;        % EMA квадрата градиента для h (RMSprop-нормализация)
        m_alpha = 0;    % EMA квадрата градиента для alpha
        beta_rms = 0.9; % Коэффициент затухания EMA
    end

    methods
        function obj = Heating2DTunableModel(h, nx, ny, lambda, alpha, Lx, Ly, T0, nt, learningRate)
            arguments
                h (1,1) double
                nx (1,1) double
                ny (1,1) double
                lambda (1,1) double
                alpha (1,1) double
                Lx (1,1) double
                Ly (1,1) double
                T0 (1,1) double
                nt (1,1) double
                learningRate (1,1) double = 0.01
            end

            obj.Lx = Lx;
            obj.Ly = Ly;
            obj.alpha = alpha;
            obj.lambda = lambda;
            obj.h = h;
            obj.nx = nx;
            obj.ny = ny;
            obj.T0 = T0;
            obj.nt = nt;
            obj.learningRate = learningRate;
        end

        function result = CalcCoreFunction(obj, InputParams)
            arguments
                obj
                InputParams (2,1) double {mustBePositive}
            end

            T_inf = InputParams(2,1);
            time = InputParams(1,1);

            dx = obj.Lx / (obj.nx - 1);
            dy = obj.Ly / (obj.ny - 1);
            dt = time / obj.nt;

            N = obj.nx * obj.ny;
            T = ones(N, 1) * obj.T0;

            theta = 0.5;
            rx = obj.alpha * dt / dx^2;
            ry = obj.alpha * dt / dy^2;

            for k = 1:obj.nt
                A = sparse(N, N);
                b = zeros(N, 1);

                for i = 1:obj.nx
                    for j = 1:obj.ny
                        idx = (j - 1) * obj.nx + i;

                        if i > 1 && i < obj.nx && j > 1 && j < obj.ny
                            A(idx, idx) = 1 + theta * (rx + ry);
                            A(idx, idx - 1) = -theta * rx / 2;
                            A(idx, idx + 1) = -theta * rx / 2;
                            A(idx, idx - obj.nx) = -theta * ry / 2;
                            A(idx, idx + obj.nx) = -theta * ry / 2;

                            b(idx) = T(idx) + (1 - theta) * rx * (T(idx - 1) - 2 * T(idx) + T(idx + 1)) / 2 ...
                                + (1 - theta) * ry * (T(idx - obj.nx) - 2 * T(idx) + T(idx + obj.nx)) / 2;

                        elseif i == 1 && j > 1 && j < obj.ny
                            A(idx, idx) = 1 + theta * rx + theta * ry / 2 + theta * rx * dx * obj.h / obj.lambda;
                            A(idx, idx + 1) = -theta * rx;
                            A(idx, idx - obj.nx) = -theta * ry / 4;
                            A(idx, idx + obj.nx) = -theta * ry / 4;

                            b(idx) = T(idx) + (1 - theta) * rx * (T(idx + 1) - T(idx) + dx * obj.h / obj.lambda * (T_inf - T(idx))) ...
                                + (1 - theta) * ry * (T(idx - obj.nx) - 2 * T(idx) + T(idx + obj.nx)) / 2 ...
                                + theta * rx * dx * obj.h / obj.lambda * T_inf;

                        elseif i == obj.nx && j > 1 && j < obj.ny
                            A(idx, idx) = 1 + theta * rx + theta * ry / 2 + theta * rx * dx * obj.h / obj.lambda;
                            A(idx, idx - 1) = -theta * rx;
                            A(idx, idx - obj.nx) = -theta * ry / 4;
                            A(idx, idx + obj.nx) = -theta * ry / 4;

                            b(idx) = T(idx) + (1 - theta) * rx * (T(idx - 1) - T(idx) + dx * obj.h / obj.lambda * (T_inf - T(idx))) ...
                                + (1 - theta) * ry * (T(idx - obj.nx) - 2 * T(idx) + T(idx + obj.nx)) / 2 ...
                                + theta * rx * dx * obj.h / obj.lambda * T_inf;

                        elseif j == obj.ny && i > 1 && i < obj.nx
                            A(idx, idx) = 1 + theta * rx / 2 + theta * ry + theta * ry * dy * obj.h / obj.lambda;
                            A(idx, idx - 1) = -theta * rx / 4;
                            A(idx, idx + 1) = -theta * rx / 4;
                            A(idx, idx - obj.nx) = -theta * ry;

                            b(idx) = T(idx) + (1 - theta) * rx * (T(idx - 1) - 2 * T(idx) + T(idx + 1)) / 2 ...
                                + (1 - theta) * ry * (T(idx - obj.nx) - T(idx) + dy * obj.h / obj.lambda * (T_inf - T(idx))) ...
                                + theta * ry * dy * obj.h / obj.lambda * T_inf;

                        elseif j == 1 && i > 1 && i < obj.nx
                            A(idx, idx) = 1 + theta * rx / 2 + theta * ry;
                            A(idx, idx - 1) = -theta * rx / 4;
                            A(idx, idx + 1) = -theta * rx / 4;
                            A(idx, idx + obj.nx) = -theta * ry;

                            b(idx) = T(idx) + (1 - theta) * rx * (T(idx - 1) - 2 * T(idx) + T(idx + 1)) / 2 ...
                                + (1 - theta) * ry * (T(idx + obj.nx) - T(idx));

                        elseif i == 1 && j == 1
                            A(idx, idx) = 1 + theta * rx + theta * ry + theta * rx * dx * obj.h / obj.lambda;
                            A(idx, idx + 1) = -theta * rx;
                            A(idx, idx + obj.nx) = -theta * ry;

                            b(idx) = T(idx) + (1 - theta) * rx * (T(idx + 1) - T(idx) + dx * obj.h / obj.lambda * (T_inf - T(idx))) ...
                                + (1 - theta) * ry * (T(idx + obj.nx) - T(idx)) ...
                                + theta * rx * dx * obj.h / obj.lambda * T_inf;

                        elseif i == obj.nx && j == 1
                            A(idx, idx) = 1 + theta * rx + theta * ry + theta * rx * dx * obj.h / obj.lambda;
                            A(idx, idx - 1) = -theta * rx;
                            A(idx, idx + obj.nx) = -theta * ry;

                            b(idx) = T(idx) + (1 - theta) * rx * (T(idx - 1) - T(idx) + dx * obj.h / obj.lambda * (T_inf - T(idx))) ...
                                + (1 - theta) * ry * (T(idx + obj.nx) - T(idx)) ...
                                + theta * rx * dx * obj.h / obj.lambda * T_inf;

                        elseif i == 1 && j == obj.ny
                            A(idx, idx) = 1 + theta * rx + theta * ry + theta * rx * dx * obj.h / obj.lambda + theta * ry * dy * obj.h / obj.lambda;
                            A(idx, idx + 1) = -theta * rx;
                            A(idx, idx - obj.nx) = -theta * ry;

                            b(idx) = T(idx) + (1 - theta) * rx * (T(idx + 1) - T(idx) + dx * obj.h / obj.lambda * (T_inf - T(idx))) ...
                                + (1 - theta) * ry * (T(idx - obj.nx) - T(idx) + dy * obj.h / obj.lambda * (T_inf - T(idx))) ...
                                + theta * rx * dx * obj.h / obj.lambda * T_inf ...
                                + theta * ry * dy * obj.h / obj.lambda * T_inf;

                        elseif i == obj.nx && j == obj.ny
                            A(idx, idx) = 1 + theta * rx + theta * ry + theta * rx * dx * obj.h / obj.lambda + theta * ry * dy * obj.h / obj.lambda;
                            A(idx, idx - 1) = -theta * rx;
                            A(idx, idx - obj.nx) = -theta * ry;

                            b(idx) = T(idx) + (1 - theta) * rx * (T(idx - 1) - T(idx) + dx * obj.h / obj.lambda * (T_inf - T(idx))) ...
                                + (1 - theta) * ry * (T(idx - obj.nx) - T(idx) + dy * obj.h / obj.lambda * (T_inf - T(idx))) ...
                                + theta * rx * dx * obj.h / obj.lambda * T_inf ...
                                + theta * ry * dy * obj.h / obj.lambda * T_inf;
                        end
                    end
                end

                T = A \ b;
            end

            T_matrix = reshape(T, obj.nx, obj.ny);
            result = mean(T_matrix, 'all');
        end

        function result = GetNumOfInputParams(obj)
            result = 2;
        end

        % ---- Методы ITunableCoreF ----

        function TuneParameters(obj, InputParams, dJ_dC)
            % Настройка h и alpha на основе градиента ошибки ∂J/∂C
            % dJ_dC — скаляр: градиент функции потерь по выходу C = CalcCoreFunction
            % Используются конечные разности + RMSprop-нормализация градиентов
            % для компенсации разницы масштабов ∂C/∂h и ∂C/∂alpha (до 5 порядков)
            arguments
                obj
                InputParams
                dJ_dC (1,1) double
            end

            epsRel = 1e-6;
            epsRMS = 1e-12;  % защита от деления на ноль в RMSprop

            % --- ∂C/∂h (центральные конечные разности) ---
            h0 = obj.h;
            delta_h = max(abs(h0) * epsRel, 1e-6);
            obj.h = h0 + delta_h;
            C_plus_h = obj.CalcCoreFunction(InputParams);
            obj.h = h0 - delta_h;
            C_minus_h = obj.CalcCoreFunction(InputParams);
            dC_dh = (C_plus_h - C_minus_h) / (2 * delta_h);
            obj.h = h0;

            % --- ∂C/∂alpha (центральные конечные разности) ---
            alpha0 = obj.alpha;
            delta_alpha = max(abs(alpha0) * epsRel, 1e-12);
            obj.alpha = alpha0 + delta_alpha;
            C_plus_a = obj.CalcCoreFunction(InputParams);
            obj.alpha = alpha0 - delta_alpha;
            C_minus_a = obj.CalcCoreFunction(InputParams);
            dC_dalpha = (C_plus_a - C_minus_a) / (2 * delta_alpha);
            obj.alpha = alpha0;

            % --- Градиенты ∂J/∂h и ∂J/∂alpha (цепное правило) ---
            grad_h = dJ_dC * dC_dh;
            grad_alpha = dJ_dC * dC_dalpha;

            % --- RMSprop-нормализация (независимо для каждого параметра) ---
            obj.m_h = obj.beta_rms * obj.m_h + (1 - obj.beta_rms) * grad_h^2;
            obj.m_alpha = obj.beta_rms * obj.m_alpha + (1 - obj.beta_rms) * grad_alpha^2;

            % --- Обновление параметров ---
            obj.h      = obj.h      - obj.learningRate * grad_h      / sqrt(obj.m_h + epsRMS);
            obj.alpha  = obj.alpha  - obj.learningRate * grad_alpha  / sqrt(obj.m_alpha + epsRMS);
        end

        function params = GetTunableParameters(obj)
            % Возвращает структуру с текущими значениями настраиваемых параметров
            params.h = obj.h;
            params.alpha = obj.alpha;
        end

        function SetTunableParameters(obj, params)
            % Устанавливает настраиваемые параметры из структуры
            arguments
                obj
                params struct
            end

            if isfield(params, 'h')
                obj.h = params.h;
            end
            if isfield(params, 'alpha')
                obj.alpha = params.alpha;
            end
        end

        function numParams = GetNumOfTunableParameters(obj)
            % Возвращает количество настраиваемых параметров
            numParams = 2;
        end

        % ---- Сериализация ----

        function data = save(obj)
            data.Lx = obj.Lx;
            data.Ly = obj.Ly;
            data.alpha = obj.alpha;
            data.lambda = obj.lambda;
            data.h = obj.h;
            data.nx = obj.nx;
            data.ny = obj.ny;
            data.T0 = obj.T0;
            data.nt = obj.nt;
            data.learningRate = obj.learningRate;
        end

        function load(obj, data)
            obj.Lx = data.Lx;
            obj.Ly = data.Ly;
            obj.alpha = data.alpha;
            obj.lambda = data.lambda;
            obj.h = data.h;
            obj.nx = data.nx;
            obj.ny = data.ny;
            obj.T0 = data.T0;
            obj.nt = data.nt;
            obj.learningRate = data.learningRate;
        end
    end

    methods (Static)
        function obj = createFromData(data)
            obj = Heating2DTunableModel(...
                data.h, data.nx, data.ny, data.lambda, data.alpha, ...
                data.Lx, data.Ly, data.T0, data.nt, data.learningRate);
        end
    end
end