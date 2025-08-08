classdef BWMatrix
    
    properties (Access = private)
        Rows BWGraph.CustomMatrix.BWRow   
    end
    
    methods
        %% Конструктор
        function obj = BWMatrix(varargin)
           
            obj.Rows = BWGraph.CustomMatrix.BWRow.empty(0,1); % Инициализация пустым массивом
            
            if nargin > 0
                if iscell(varargin{1})
                    % Создание из cell-массива
                    for i = 1:length(varargin{1})
                        obj = obj.addRow(varargin{1}{i});
                    end
                else
                    % Создание из JaggedRow объектов
                    for i = 1:nargin
                        if isa(varargin{i}, 'BWGraph.CustomMatrix.BWRow')
                            obj.Rows(end+1) = varargin{i};
                        else
                            error('Аргументы должны быть BWRow или cell-массивом');
                        end
                    end
                end
            end
        end
        
        %% Добавление строки
        function obj = addRow(obj, rowData)
            %   addRow Добавляет новую строку в матрицу
            %   rowData - массив данных для новой строки
            
            newRow = BWGraph.CustomMatrix.BWRow(rowData);
            obj.Rows(end+1) = newRow;
        end
        
        %% Получение строки
        function row = getRow(obj, rowIndex)
            %getRow Возвращает строку по индексу
            row = obj.Rows(rowIndex).Data;
        end
        
        %% Получение элемента
        function element = getElement(obj, rowIndex, colIndex)
            %getElement Возвращает элемент по индексам
            row = obj.Rows(rowIndex).Data;
            element = row(colIndex);
        end
        
        %% Изменение элемента
        function obj = setElement(obj, rowIndex, colIndex, value)
            %setElement Изменяет значение элемента
            obj.Rows(rowIndex).Data(colIndex) = value;
        end
        
        %% Количество строк
        function count = rowCount(obj)
            %rowCount Возвращает количество строк
            count = numel(obj.Rows);
        end
        
        %% Длина строки
        function len = rowLength(obj, rowIndex)
            %rowLength Возвращает длину указанной строки
            len = length(obj.Rows(rowIndex).Data);
        end
        
        %% Визуализация матрицы
        function display(obj)
            %display Выводит матрицу в консоль
            fprintf('Matrix have (%d rows):\n', obj.rowCount());
            for i = 1:obj.rowCount()
                fprintf('Row %d: ', i);
                fprintf('%g ', obj.getRow(i));
                fprintf('\n');
            end
        end
        
        %% Конвертация в cell-массив
        function cellArray = toCell(obj)
            %toCell Конвертирует в cell-массив
            cellArray = cell(1, obj.rowCount());
            for i = 1:obj.rowCount()
                cellArray{i} = obj.getRow(i);
            end
        end
    end
end
