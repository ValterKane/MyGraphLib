classdef Edge < handle
    properties
        SourceNode BWGraph.Node
        TargetNode BWGraph.Node
        Alfa {mustBeFinite}
        Beta {mustBeFinite}
    end

    methods
        function obj = Edge(source, target, alfa, beta)
            obj.SourceNode = source;
            obj.TargetNode = target;
            obj.Alfa = alfa;
            obj.Beta = beta;
        end
    end
end

