classdef SimpleRandomGen < BWGraph.RandomGenerator.IRandomGen
    properties
        stat = 0.1;
    end
    methods
        function obj = SimpleRandomGen(stat)
            obj.stat = stat;
        end

        function result = Generate(obj)
            result = obj.stat * (0.9+0.2*rand());
        end
    end
end

