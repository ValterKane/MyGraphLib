classdef ContextProjector < coreFunctions.ICoreF
    % ContextProjector — ядровая функция без собственного решателя.
    % При K=1: пассивный узел (выход ≡ 0 для linear, ≡ 0.5 для sigmoid).
    % При K>1: пропускает через себя контекст от входящих соседей,
    % масштабированный через Gamma и функцию активации.
    %
    % Использование:
    %   proj = Node(id, 1, 'Black', coreFunctions.ContextProjector(), 'sigmoid');
    %
    % Топология: A(решатель) → Proj(контекст) → B(решатель) → ...

    methods
        function tf = SupportsContext(~)
            tf = true;
        end

        function result = CalcCoreFunction(~, InputParams)
            arguments
                ~
                InputParams (:,1) double
            end
            if isempty(InputParams)
                result = 0;
            else
                result = InputParams(1);
            end
        end

        function n = GetNumOfInputParams(~)
            n = 1;  % Dummy: совместимость с BWMatrix (реальный вход игнорируется)
        end

        function aug = AugmentInput(~, ~, ctx_vec)
            % Игнорирует baseInput, возвращает вектор контекста
            aug = ctx_vec;
        end

        function df = CalcContextDerivative(~, ~, ctx_vec)
            % d(Core_j)/d(ctx_j) = 1 (identity), остальные 0
            df = eye(length(ctx_vec));
        end

        function data = save(~)
            data = struct();
        end

        function load(~, ~)
        end
    end

    methods (Static)
        function obj = createFromData(~)
            obj = coreFunctions.ContextProjector();
        end
    end
end
