classdef PlateHeatingModel < coreFunctions.ICoreF
    properties
        L;          % Толщина пластины [м]
        alpha;      % Коэффициент температуропроводности [м^2/с]
        lambda;     % Теплопроводность материала [Вт/(м·K)]
        h;          % Коэффициент теплоотдачи [Вт/(м^2·K)]
        nx;         % Количество узлов по пространству
        T0;
        nt
    end
    
    methods
        function obj = PlateHeatingModel(h, nx, lambda, alpha, L, T0, nt)
            % Конструктор класса с параметрами по умолчанию
            obj.L = L;
            obj.alpha = alpha;
            obj.lambda = lambda;
            obj.h = h;
            obj.nx = nx;
            obj.T0 = T0;
            obj.nt = nt;
        end
        
        function result = CalcCoreFunction(obj, InputParams)
            arguments
                obj 
                InputParams (2,1) double {mustBePositive}
            end
            
            % Извлечение параметров
            T_inf = InputParams(2,1);
            time = InputParams(1,1);
            a = 0;
            
            % Расчет шагов
            dx = obj.L/(obj.nx-1);
            dt = time/obj.nt;
            
            % Инициализация температурного поля
            T = ones(obj.nx,1)*obj.T0;
            T_new = T;
            
            % Основной цикл по времени
            for k = 1:obj.nt

                % Внутренние точки (явная схема)
                for i = 2:obj.nx-1
                    T_new(i) = T(i) + obj.alpha*dt/dx^2 * (T(i+1) - 2*T(i) + T(i-1));
                end

                % Граничные условия III рода
                % Левая граница (x=0): -lambda*dT/dx = h*(T_inf - T)
                T_new(1) = T(1) + obj.alpha*dt/dx^2 * (2*T(2) - 2*T(1) + 2*dx*obj.h/obj.lambda*(T_inf - T(1)));

                % Правая граница (x=L): lambda*dT/dx = h*(T_inf - T)
                T_new(obj.nx) = T(obj.nx) + obj.alpha*dt/dx^2 * (2*T(obj.nx-1) - 2*T(obj.nx) + 2*dx*obj.h/obj.lambda*(T_inf - T(obj.nx)));
                
                

                T = T_new;
                
            end
            a = mean(T);
            % Формирование результата
            result = mean(T);
        end
        
        function result = GetNumOfInputParams(obj)
            % Возвращает количество входных параметров
            result = 2; % T_inf, time
        end
        
        function data = save(obj)
            % Сериализация объекта
            data.L = obj.L;
            data.alpha = obj.alpha;
            data.lambda = obj.lambda;
            data.h = obj.h;
            data.nx = obj.nx;
        end
        
        function load(obj, data)
            % Десериализация объекта
            obj.L = data.L;
            obj.alpha = data.alpha;
            obj.lambda = data.lambda;
            obj.h = data.h;
            obj.nx = data.nx;
        end
    end
    
    methods (Static)
        function obj = createFromData(data)
            % Создание объекта из данных
            obj = PlateHeatingModel();
            obj.load(data);
        end
    end
end
