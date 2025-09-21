% Пример комплексной функции теплового баланса
classdef HeatTransferBC < coreFunctions.ICoreF
    % Класс для расчета граничных условий 3-го рода (смешанных)
    % с учетом конвекции и излучения

    properties
        h           % Коэффициент конвективной теплоотдачи [W/(m²·K)]
        epsilon     % Степень черноты поверхности [0..1]
        sigma = 5.670374419e-8 % Постоянная Стефана-Больцмана [W/(m²·K⁴)]
    end

    methods
        function obj = HeatTransferBC(h, epsilon)
            % Конструктор класса
            % Входные параметры:
            % h - коэффициент конвективной теплоотдачи
            % epsilon - степень черноты поверхности

            obj.h = h;
            obj.epsilon = epsilon;
        end

        function result = CalcCoreFunction(obj, InputParams)
            arguments
                obj
                InputParams (3,1) double {mustBePositive}
            end

            T_guess = InputParams(1,1);
            T_Inf = InputParams(2,1);
            T_Surround = InputParams(3,1);

            heatBalance = @(Ts) obj.h*(Ts-T_Inf) + obj.sigma*obj.epsilon*(Ts^4 - T_Surround^4);

            result = fzero(heatBalance, T_guess);
        end

        function  result = GetNumOfInputParams(obj)
            result = 3;
        end

        function data = save(obj)
            % Сериализация объекта в структуру
            data = struct();
            data.ClassName = class(obj);
            data.h = obj.h;
            data.epsilon = obj.epsilon;
            data.sigma = obj.sigma;
        end
        
        function load(obj, data)
            % Десериализация объекта из структуры
            if ~strcmp(data.ClassName, class(obj))
                error('Class mismatch during loading');
            end
            obj.h = data.h;
            obj.epsilon = data.epsilon;
            obj.sigma = data.sigma;
        end
 
    end

    methods (Static)
        function obj = createFromData(data)
            % Фабричный метод для создания экземпляра при загрузке
            obj = HeatTransferBC();
            obj.load(data);
        end
    end
end