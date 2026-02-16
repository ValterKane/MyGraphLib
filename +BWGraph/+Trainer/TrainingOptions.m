classdef TrainingOptions
        
    properties
        % Основные параметры оптимизатора
        LearningRate    (1,1) double {mustBePositive, mustBeFinite} = 0.001
        Beta1           (1,1) double {mustBePositive, mustBeFinite} = 0.9
        Beta2           (1,1) double {mustBePositive, mustBeFinite} = 0.999
        Eps             (1,1) double {mustBePositive, mustBeFinite} = 1e-8
        
        % Параметры вершин
        NodeWeight      (1,:) double = []
        NodeSize        (1,:) double = []

        % Параметры батча
        BatchSize       (1,1) double {mustBePositive, mustBeFinite} = 1;
        
        % Параметры обучения
        Epoches         (1,1) double {mustBePositive, mustBeInteger} = 100
        TargetError     (1,1) double {mustBePositive, mustBeFinite} = 1e-5
        
        % Параметры градиента
        ClipUp          (1,1) double {mustBeFinite} = 1e5
        ClipDown        (1,1) double {mustBeFinite} = -1e5
        
        % Параметры регуляризации
        Lambda_Alph          (1,1) double {mustBePositive} = 0.01
        Lambda_Beta          (1,1) double {mustBePositive} = 0.01
        Lambda_Gamma          (1,1) double {mustBePositive} = 0.01
        Lambda_Agg           (1,1) double {mustBeNonnegative} = 0

        % Параметры функций настройки
        HuberDelta (1,1) double {mustBePositive, mustBeFinite} = 1
        
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
                options.NodeWeight      (1,:) double {mustBeFinite}
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
                options.ErrorMetric     (1,1) string
                options.LossFunction    (1,1) string
            end
            
            % Применяем переданные значения
            if nargin > 0
                fields = fieldnames(options);
                for i = 1:length(fields)
                    field = fields{i};
                    if ~isempty(options.(field))
                        obj.(field) = options.(field);
                    end
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

            if isfield(options, 'NodeWeight')
                obj.NodeWeight = options.NodeWeight;
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

