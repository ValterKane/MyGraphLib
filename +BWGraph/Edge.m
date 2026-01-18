classdef Edge < handle
    properties
        SourceNode BWGraph.Node
        TargetNode BWGraph.Node
        Alfa {mustBeFinite}
        Beta {mustBeFinite}
        Gamma  {mustBeFinite}
        Delta  {mustBeFinite}
        ID {mustBeFinite}
    end

    methods
        function obj = Edge(SourceNode, TargetNode, Alfa, Beta, Gamma, Delta, Id)
            if nargin < 6
                Delta = 0;
            end
            if nargin < 5
                Gamma = 0;
            end
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
            obj.Gamma = Gamma;
            obj.Delta = Delta;
            obj.ID = Id;
        end
    end
end

