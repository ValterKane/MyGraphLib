classdef EvolutionModelShell < handle % Нестационарная модель
    
    properties (Access = private)
        A (:,:,:) {mustBeInteger} % Тензор А смежности графовой модели;
        alfa (:,:,:)              % Тензор для альфа-значений;
        beta (:,:,:)              % Тензор для бета-значений;
        teta (:,:)                % Матрица тета-значений;
        F (:,:)                   % Матрица значений модели;
        CoreFunction              % Функция ядра;
        fi (:,:)                  % Матрица ядровых функций;
        dT {mustBeFinite}
    end

    methods( Access = public)
    
        % Конструктор
        function obj = EvolutionModelShell(AdjArray, initial, CoreFunction)
            arguments
                AdjArray 
                initial 
                CoreFunction IBoundary
            end
            % Инициализация основной структуры модели
            obj.A = AdjArray;
            obj.CoreFunction = CoreFunction;


            % Инициализируем массивы начальных значений
            [rows, cols, layers] = size(obj.A);

            obj.alfa = zeros(rows,cols, layers);
            obj.beta = zeros(rows,cols, layers);
            obj.teta = zeros(layers, cols);
            obj.F = zeros(layers,cols);
            obj.fi = zeros(layers, cols);

            for n = 1:layers
                for i = 1:rows
                    for j = 1:cols
                        if obj.A(i,j,n) == 1
                            obj.alfa(i,j,n) = rand();
                            obj.beta(i,j,n) = randi([25,100]) * rand();
                            obj.F(n,j) = initial;
                            obj.teta(n,j) = randi([1,10]) * rand();
                        end
                    end
                end
            end
        end
    
        function result = Forward(obj, XValues, dT)
            arguments
                obj EvolutionModelShell
                XValues (:,:,:) double {mustBeFinite}   
                dT {mustBeFinite}
            end

            if size(obj.F, 2) ~= size(XValues, 2)
                error("Ошибка! Количество ожидаемых значений должно быть равно" + ...
                    " количеству вершин модели!")
            elseif size(obj.F,1) ~= size(XValues, 3)
                error("Для прямого расчета необходимо знать условия для " + ...
                    "всех внутренних слоев!")
            end
            

            %  --- Алгоритм расчета по слоям ---
            for n = 2:size(obj.F, 1)
                % --- Алгоритм расчета внутри слоя ---
                for i = 1:size(obj.F,2)
                    % Вычисление функции в ядре
                    obj.fi(n-1, i) = obj.CoreFunction.CalcCoreFunction(XValues(:, i, n-1));
                    
                    % --- Инициализация вспомогательных функций ---
                    g1 = 0;
                    g2 = 0;
                    g3 = 0;

                    % --- Внутренний цикл расчета ---
                    for j = 1:size(obj.A,1)
                        g1 = g1 + (obj.F(n-1,j) * obj.alfa(j,i,n-1) + obj.beta(j,i,n-1)) * obj.A(j,i,n-1);
                        g2 = g2 + obj.A(i,j,n-1) * obj.alfa(i,j,n-1);
                        g3 = g3 + obj.A(i,j,n-1) * obj.beta(i,j,n-1);
                    end
                    
                    obj.F(n,i) = obj.teta(n-1,i) * (obj.fi(n-1,i) + dT + g1 + dT*g2 - g3) / (g2+1);
                end
            end

            result = obj.F;
        end
        
        function res = getNumOfVertex(obj)
            res = size(obj.A,1);
        end

        function res = getNumOfParam(obj)
            res = obj.CoreFunction.GetNumOfInputParams();
        end

        function res = getNumOfLayers(obj)
            res = size(obj.A,3);
        end
    end

end