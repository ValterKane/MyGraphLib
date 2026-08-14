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
        % Раздельный клиппинг для разных типов параметров (если пусто — используется общий ClipUp/ClipDown или авто-калибровка)
        ClipUp_Alpha    double = []
        ClipDown_Alpha  double = []
        ClipUp_Beta     double = []
        ClipDown_Beta   double = []
        ClipUp_Gamma    double = []
        ClipDown_Gamma  double = []
        % Автоматический подбор границ клиппинга по первому батчу
        AutoCalibrateClip (1,1) logical = false
        ClipPercentile    (1,1) double {mustBePositive, mustBeFinite, mustBeLessThanOrEqual(ClipPercentile, 100)} = 95
        
        % Параметры регуляризации
        Lambda_Alph          (1,1) double {mustBeNonnegative}
        Lambda_Beta          (1,1) double {mustBeNonnegative}
        Lambda_Gamma         (1,1) double {mustBeNonnegative}
        Lambda_Agg           (1,1) double {mustBeNonnegative}
        Lambda_Self          (1,1) double {mustBeNonnegative} = 0
        Lambda_Struct        (1,1) double {mustBeNonnegative} = 1
        Lambda_Stability (1,1) double {mustBeNonnegative} = 0
        RpShiftPercent   (1,1) double {mustBeNonnegative} = 20
        StructInitAlpha  (1,1) double {mustBeNonnegative} = 0.5
        StructInitBeta   (1,1) double = 1
        EnablePlateauEscape (1,1) logical = true
        StructuralCooldown (1,1) double {mustBeNonnegative, mustBeInteger} = 2
        StructuralComplexityPenalty (1,1) double {mustBeNonnegative} = 0
        AlphaMin          (1,1) double {mustBeNonnegative} = 0.01
        AlphaSafetyFactor (1,1) double {mustBePositive} = 0.8
        StabilityClampFactor (1,1) double {mustBePositive} = 0.99
        StructuralCleanupThreshold (1,1) double {mustBeNonnegative} = 0.01
        LRDecayInterval  (1,1) double {mustBePositive, mustBeInteger} = 1
        ContextStages        (1,1) double {mustBePositive, mustBeInteger} = 1

        % Параметры функций настройки
        HuberDelta (1,1) double {mustBePositive, mustBeFinite} = 1;
        
        % Параметры анализа
        TargetNodeIndices (1,:) double = [] % По умолчанию анализируются все белые вершины
        ErrorMetric     (1,1) string {mustBeMember(ErrorMetric, {'mae', 'mse', 'rmse', 'mape'})} = "mae"
        LossFunction    (1,1) string {mustBeMember(LossFunction, {'mae', 'mse', 'huber', 'logcosh'})} = "mae"

        % Параметры структурного поиска (жадный NAS)
        EnableStructuralSearch (1,1) logical = false     % Включить/выключить оптимизацию топологии
        StructuralSearchInterval (1,1) double {mustBePositive, mustBeInteger} = 50  % Каждые N эпох
        StructuralSearchCandidates (1,1) double {mustBePositive, mustBeInteger} = 10 % Кандидатов на поиск
        StructuralSearchEpochs (1,1) double {mustBePositive, mustBeInteger} = 5      % Эпох быстрой настройки кандидата
        StructuralSearchMaxEdges (1,1) double {mustBeNonnegative} = inf              % Макс. рёбер
        StructuralSearchMinEdges (1,1) double {mustBeNonnegative} = 0                % Мин. рёбер
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
                options.ClipUp_Alpha    (1,1) double {mustBeFinite}
                options.ClipDown_Alpha  (1,1) double {mustBeFinite}
                options.ClipUp_Beta     (1,1) double {mustBeFinite}
                options.ClipDown_Beta   (1,1) double {mustBeFinite}
                options.ClipUp_Gamma    (1,1) double {mustBeFinite}
                options.ClipDown_Gamma  (1,1) double {mustBeFinite}
                options.AutoCalibrateClip (1,1) logical
                options.ClipPercentile  (1,1) double {mustBePositive, mustBeFinite, mustBeLessThanOrEqual(options.ClipPercentile, 100)}
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
                options.Lambda_Stability (1,1) double {mustBeNonnegative}
                options.RpShiftPercent   (1,1) double {mustBeNonnegative}
                options.StructInitAlpha  (1,1) double {mustBeNonnegative}
                options.StructInitBeta   (1,1) double
                options.EnablePlateauEscape (1,1) logical
                options.StructuralCooldown (1,1) double {mustBeNonnegative, mustBeInteger}
                options.StructuralComplexityPenalty (1,1) double {mustBeNonnegative}
                options.AlphaMin          (1,1) double {mustBeNonnegative}
                options.AlphaSafetyFactor (1,1) double {mustBePositive}
                options.StabilityClampFactor (1,1) double {mustBePositive}
                options.StructuralCleanupThreshold (1,1) double {mustBeNonnegative}
                options.LRDecayInterval (1,1) double {mustBePositive, mustBeInteger} = 1
                options.Lambda_Self     (1,1) double {mustBeNonnegative}
                options.EnableStructuralSearch (1,1) logical
                options.StructuralSearchInterval (1,1) double {mustBePositive, mustBeInteger}
                options.StructuralSearchCandidates (1,1) double {mustBePositive, mustBeInteger}
                options.StructuralSearchEpochs (1,1) double {mustBePositive, mustBeInteger}
                options.StructuralSearchMaxEdges (1,1) double {mustBeNonnegative}
                options.StructuralSearchMinEdges (1,1) double {mustBeNonnegative}
                options.ContextStages (1,1) double {mustBePositive, mustBeInteger} = 1
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

        function n = getNodeMultiplier(obj, idx)
            % Возвращает множитель скорости обучения для вершины idx
            % Если NodeSize не задан — по умолчанию 1 для всех вершин
            if isempty(obj.NodeSize)
                n = 1;
            else
                n = obj.NodeSize(idx);
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

