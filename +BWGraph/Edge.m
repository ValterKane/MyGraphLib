classdef Edge < handle
    properties
        SourceNode BWGraph.Node
        TargetNode BWGraph.Node
        Alfa {mustBeFinite}
        Beta {mustBeFinite}
        ID {mustBeFinite}
    end

    methods
        function obj = Edge(SourceNode, TargetNode, Alfa, Beta, Id)
            if nargin < 4
                Beta = 0;
            end
            if nargin < 3
                Alfa = 0;
            end
            
            obj.SourceNode = SourceNode;
            obj.TargetNode = TargetNode;
            obj.Alfa = Alfa;
            obj.Beta = Beta;
            obj.ID = Id;
        end
    end
end

