classdef (Abstract) ICoreF
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
end    