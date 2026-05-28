classdef TrainingOptions < handle
        
    properties
        % Основные параметры оптимизатора
        LearningRate    (1,1) double {mustBePositive, mustBeFinite} = 0.01;
        Beta1           (1,1) double {mustBePositive, mustBeFinite} = 0.9;
        Beta2           (1,1) double {mustBePositive, mustBeFinite} = 0.999;
        Eps             (1,1) double {mustBePositive, mustBeFinite} = 1e-8;
        
        % Параметры вершин
        NodeSize        (1,:) double = []

        % Параметры батча
        BatchSize       (1,1) double {mustBePositive, mustBeFinite} = 1;
        
        % Параметры обучения
        Epoches         (1,1) double {mustBePositive, mustBeInteger} = 1;
        TargetError     (1,1) double {mustBePositive, mustBeFinite} = 0.5;
        
        % Параметры градиента
        ClipUp          (1,1) double {mustBeFinite}
        ClipDown        (1,1) double {mustBeFinite}
        
        % Параметры регуляризации
        Lambda_Alph          (1,1) double {mustBeNonnegative}
        Lambda_Beta          (1,1) double {mustBeNonnegative}
        Lambda_Gamma         (1,1) double {mustBeNonnegative}
        Lambda_Agg           (1,1) double {mustBeNonnegative}
        Lambda_Self          (1,1) double {mustBeNonnegative}
        Lambda_Struct        (1,1) double {mustBeNonnegative}

        % Параметры функций настройки
        HuberDelta (1,1) double {mustBePositive, mustBeFinite} = 1;
        
        % Параметры анализа
        TargetNodeIndices (1,:) double = [] % По умолчанию анализируются все белые вершины
        ErrorMetric     (1,1) string {mustBeMember(ErrorMetric, {'mae', 'mse', 'rmse', 'mape'})} = "mae"
        LossFunction    (1,1) string {mustBeMember(LossFunction, {'mae', 'mse', 'huber', 'logcosh'})} = "mae"
    end
    
    methods
        function obj = TrainingOptions(options)
            % TrainingConfig Конструктор класса
            %   Принимает именованные аргументы для инициализации свойств
            
            arguments
                options.LearningRate    (1,1) double {mustBePositive, mustBeFinite}
                options.Beta1           (1,1) double {mustBePositive, mustBeFinite}
                options.Beta2           (1,1) double {mustBePositive, mustBeFinite}
                options.Eps             (1,1) double {mustBePositive, mustBeFinite}
                options.NodeSize        (1,:) double {mustBeFinite}
                options.Epoches         (1,1) double {mustBePositive, mustBeInteger}
                options.ClipUp          (1,1) double {mustBeFinite}
                options.ClipDown        (1,1) double {mustBeFinite}
                options.TargetError     (1,1) double {mustBePositive, mustBeFinite}
                options.Lambda_Alph          (1,1) double {mustBePositive}
                options.Lambda_Beta          (1,1) double {mustBePositive}
                options.Lambda_Gamma          (1,1) double {mustBePositive}
                options.HuberDelta      (1,1) double {mustBePositive, mustBeFinite}
                options.Lambda_Agg      (1,1) double {mustBeNonnegative}
                options.TargetNodeIndices (1,:) double
                options.ErrorMetric     (1,1) string {mustBeMember(options.ErrorMetric, {'mae', 'mse', 'rmse', 'mape'})}
                options.LossFunction    (1,1) string {mustBeMember(options.LossFunction, {'mae', 'mse', 'huber', 'logcosh'})}
                options.BatchSize       (1,1) double {mustBeFinite, mustBePositive}
                options.Lambda_Struct   (1,1) double {mustBeNonnegative}
                options.Lambda_Self     (1,1) double {mustBeNonnegative}
            end
            
            % Применяем переданные значения
            fields = fieldnames(options);
            for i = 1:length(fields)
                field = fields{i};
                if ~isempty(options.(field))
                    obj.(field) = options.(field);
                end
            end

            if isfield(options, 'ErrorMetric')
                obj.ErrorMetric = options.ErrorMetric;
            end

            if isfield(options, 'LossFunction')
                obj.LossFunction = options.LossFunction;
            end
            
            if isfield(options, 'NodeSize')
                obj.NodeSize = options.NodeSize;
            end
        end

        function config = toStruct(obj)
            % toStruct Преобразует объект в структуру
            config = struct();
            props = properties(obj);
            for i = 1:length(props)
                config.(props{i}) = obj.(props{i});
            end
        end
    end
end

