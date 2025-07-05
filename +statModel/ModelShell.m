classdef ModelShell < handle
    % Класс-оболочка графовой модели (стационарный вариант)

    properties (Access = private)
        A (:,:) {mustBeInteger} % Матрица А смежности графовой модели;
        alfa                    % Матрица для альфа-значений;
        beta                    % Матрица для бета-значений;
        F                       % Вектор значений модели;
        CoreFunction            % Функция ядра;
        fi                      % Вектор ядровых функций;
    end

    methods
        function obj = ModelShell(AdjArray, initial, CoreFunction)
            arguments
                AdjArray 
                initial 
                CoreFunction coreFunctions.IBoundary
            end
            % Инциализируем данные
            obj.A = AdjArray;
            obj.CoreFunction = CoreFunction;
            % Инициализируем массивы начальных значений
            num_vertex = size(obj.A,1);
            obj.F = zeros(1,num_vertex);
            obj.fi = zeros(1, num_vertex);

            deg = sum(obj.A,2);
            avg_deg = mean(deg);
            obj.alfa = zeros(num_vertex);
            obj.beta = zeros(num_vertex);
            for i = 1:num_vertex
                obj.F(1, i) = initial;
                for j = 1:num_vertex
                    if obj.A(i,j) == 1 && i~=j
                        % Задаем вес, обратнопропорциональный степени
                        % вершины
                        obj.alfa(i,j) = 0.1 * (avg_deg / deg(i)) * (0.9+0.2*rand());
                        % Выполним инициализацию бета, как смещений относительно
                        % начальных величин
                        obj.beta(i,j) = randi([-50,50]) * (0.95 + 0.1 * rand());
                    end
                end
            end

            % Обеспечим физическую согласованность
            obj.beta = max(min(obj.beta, 50), -50); 
        end
   
    end

    methods (Access = public)
        function Forward(obj, XValues)
            arguments
                obj statModel.ModelShell                
                XValues (:,:) double {mustBeFinite}
            end

            if size(obj.F, 2) ~= size(XValues, 2)
                error("Ошибка! Несогласованный размер входов модели")
            end
            
            % Алгоритм прямого расчета оболочкой
            for i = 1:size(obj.F, 2)
                % Внутренняя инициализация
                g_1 = 0;
                g_2 = 0;
                g_3 = 0;

                % Функция в ядре
                obj.fi(1, i) = obj.CoreFunction.CalcCoreFunction(XValues(:, i));

                % Вспомогательные функции
                for j = 1:size(obj.A, 2)
                    g_1 = g_1 + (obj.F(1,j) * obj.alfa(j,i) + obj.beta(j,i)) * obj.A(j,i);
                    g_2 = g_2 + (obj.A(i,j) * obj.beta(i,j));
                    g_3 = g_3 + (obj.A(i,j) * obj.alfa(i,j));
                end

                % Общий расчет и обновление F
                obj.F(1, i) = (obj.fi(1,i) + g_1 - g_2 ) / (g_3 + 1);
            end


        end

        function alfa = GetAlfas(obj)
            alfa = obj.alfa;
        end

        function beta = GetBetas(obj)
            beta = obj.beta;
        end

        function result = GetResult(obj)
            result = obj.F;
        end

        function result = GetResultByIndex(obj, index)
            result = obj.F(1,index);
        end

        function SetAlfaBeta(obj ,alfa, beta)
            arguments
                obj statModel.ModelShell 
                alfa (:,:) double 
                beta (:,:) double
            end

            obj.alfa = alfa;
            obj.beta = beta;
        end

        function res = GetNumOfVertex(obj)
            res = size(obj.A,2);
        end

        function res = GetNumOfParameters(obj)
            res = obj.CoreFunction.GetNumOfInputParams();
        end

        % Расчет частных производных
        function res = dFdAl(obj, i)
            g_1 = 0;
            g_2 = 0;
            g_3 = 0;
            g_4 = 0;
            for j = 1:size(obj.A,2)
               g_1 = g_1 + obj.A(i,j) * sign(obj.alfa(i,j));
               g_2 = g_2 + (obj.F(1,j)*obj.alfa(j,i) + obj.beta(j,i))*obj.A(j,i);
               g_3 = g_3 + (obj.A(i,j) * obj.beta(i,j));
               g_4 = g_4 + (obj.A(i,j) * obj.alfa(i,j));
            end
                
            res = - ((g_1 * (obj.fi(1,i) + g_2 - g_3)) / (g_4+1)^2);
        end

        function res = dFdBt(obj, i)
            g_1 = 0;
            g_2 = 0;
            
            for j = 1:size(obj.A,2)
                g_1 = g_1 + obj.A(i,j) * sign(obj.beta(i,j));
                g_2 = g_2 + obj.A(i,j) * obj.alfa(i,j);
            end

            res = - g_1 / (g_2 + 1);
        end

        
    end

end