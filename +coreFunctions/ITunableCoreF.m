classdef (Abstract) ITunableCoreF < coreFunctions.ICoreF
    methods (Abstract)
        % Метод настройки параметров на основе градиента ошибки
        % InputParams — входные параметры функции (данные для конкретной вершины)
        % dJ_dC — скаляр: градиент функции потерь по выходу ядровой функции ∂J/∂C
        % Каждая реализация сама распределяет dJ_dC по своим внутренним параметрам
        TuneParameters(obj, InputParams, dJ_dC)

        % Метод получения текущих настраиваемых параметров
        % Возвращает структуру с параметрами и их значениями
        params = GetTunableParameters(obj)

        % Метод установки настраиваемых параметров
        % params - структура с параметрами и их значениями
        SetTunableParameters(obj, params)

        % Метод получения количества настраиваемых параметров
        numParams = GetNumOfTunableParameters(obj)
    end
end