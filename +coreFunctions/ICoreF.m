classdef (Abstract) ICoreF < handle
    methods (Abstract)
        % Основные методы интерфейса
        result = CalcCoreFunction(obj, InputParams)
        result = GetNumOfInputParams(obj)

        % Методы сериализации
        data = save(obj)       % Сериализация объекта в структуру
        load(obj, data)        % Десериализация объекта из структуры
    end

    methods (Static, Abstract)
        % Метод для создания экземпляра класса при загрузке
        obj = createFromData(data)
    end

    methods
        function tf = SupportsContext(~)
            % Может ли солвер использовать контекст от предыдущего этапа Forward
            tf = false;
        end

        function augInput = AugmentInput(~, baseInput, context)
            % Расширяет входной вектор контекстом от предыдущего этапа.
            % context — вектор F-значений входящих соседей (size = nIncoming).
            % По умолчанию: контекст игнорируется.
            % Переопределяется в солверах, поддерживающих связанные процессы.
            augInput = baseInput;
        end

        function df_dctx = CalcContextDerivative(obj, baseInput, ctx_vec)
            % Производная CalcCoreFunction по вектору контекста.
            % ctx_vec — вектор F-значений входящих соседей (size = nIncoming).
            % Возвращает вектор-строку градиентов ∂Core/∂ctx_j.
            % Default: центральная конечная разность по каждому элементу.
            nCtx = length(ctx_vec);
            df_dctx = zeros(1, nCtx);
            for j = 1:nCtx
                epsVal = max(1e-4 * max(1, abs(ctx_vec(j))), 1e-6);
                ctxPlus = ctx_vec; ctxPlus(j) = ctxPlus(j) + epsVal;
                ctxMinus = ctx_vec; ctxMinus(j) = ctxMinus(j) - epsVal;
                augPlus  = obj.AugmentInput(baseInput, ctxPlus);
                augMinus = obj.AugmentInput(baseInput, ctxMinus);
                df_dctx(j) = (obj.CalcCoreFunction(augPlus) - obj.CalcCoreFunction(augMinus)) / (2 * epsVal);
            end
        end
    end
end    