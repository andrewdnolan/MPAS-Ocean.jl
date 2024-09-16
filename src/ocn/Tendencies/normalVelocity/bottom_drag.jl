"""
methods for calculating tendencies from explicit bottom drag using KA
"""

#= TO DO: Using type dispatching for BottomDrag implementation
#         selection, and confugure the BottomDrag type higherup 
#         in an `init` function that does all the config parsing
abstract type AbstractBottomDrag end

struct NoBottomDrag <: BottomDrag end

struct ExplicitBottomDrag{F} <: AbstractBottomDrag
    dragCoeff::F
end

function compute_bottom_drag_tendency!(Tend::TendencyVars,
                                       Prog::PrognosticVars,
                                       Diag::DiagnosticVars,
                                       Mesh::Mesh;
                                       Drag::NoBottomDrag,
                                       backend = KA.CPU())
    nothing
end

function compute_bottom_drag_tendency!(Tend::TendencyVars,
                                       Prog::PrognosticVars,
                                       Diag::DiagnosticVars,
                                       Mesh::Mesh;
                                       Drag::ExplicitBottomDrag,
                                       backend = KA.CPU())

    @unpack dragCoefficient = Drag

    @unpack HorzMesh, VertMesh = Mesh    
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    @unpack maxLevelEdge = VertMesh 
    @unpack nEdges, cellsOnEdge = Edges

    # unpack the normal velocity tendency term
    @unpack tendNormalVelocity = Tend 
    # get needed fields from diagnostics structure
    @unpack kineticEnergyCell = Diag
    
    # TO DO: Don't use `@view` b/c not performant on GPU's
    # get the current timelevel of prognostic variables
    normalVelocity = @view Prog.normalVelocity[:,:,end]
    layerThickness = @view Prog.layerThickness[:,:,end]

    # initialize the kernel
    kernel! = explicit_bottom_drag_tendency_kernel!(backend)
    # use kernel to compute explicit bottom drg
    kernel!(tendNormalVelocity,
            kineticEnergyCell,
            normalVelocity,
            layerThickness,
            cellsOnEdge,
            maxLevelEdge.Top, 
            dragCoefficient,
            nEdges,
            ndrange = nEdges)
    # sync the backend 
    KA.synchronize(backend)

    # pack the tendecy pack into the struct for further computation
    @pack! Tend = tendNormalVelocity
end
=#

function bottom_drag_tendency!(Tend::TendencyVars, 
                               Prog::PrognosticVars,
                               Diag::DiagnosticVars, 
                               Mesh::Mesh, 
                               Config::GlobalConfig;
                               backend = KA.CPU())
    
    bottomDragConfig = ConfigGet(Config.namelist, "bottom_drag")
    bottomDragType = ConfigGet(bottomDragConfig, "config_bottom_drag_mode")
    
    if bottomDragType == "explicit"
        dragCoeff = ConfigGet(bottomDragConfig, "config_explicit_bottom_drag_coeff")
        
        explicit_bottom_drag_tendency!(
            Tend, Prog, Diag, Mesh; dragCoefficient=dragCoeff, backend=backend
           )
    # TO DO: Add "nothing" bottom drag to inertial gravity wave config
    elseif bottomDragType == "nothing"
        continue
    else
        error("Unsupported bottom drag type") 
    end

 
end

function explicit_bottom_drag_tendency!(Tend::TendencyVars,
                                        Prog::PrognosticVars,
                                        Diag::DiagnosticVars,
                                        Mesh::Mesh;
                                        dragCoefficient = 0.001,
                                        backend = KA.CPU())

    @unpack HorzMesh, VertMesh = Mesh    
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    @unpack maxLevelEdge = VertMesh 
    @unpack nEdges, cellsOnEdge = Edges

    # unpack the normal velocity tendency term
    @unpack tendNormalVelocity = Tend 
    # get needed fields from diagnostics structure
    @unpack kineticEnergyCell = Diag
    
    # TO DO: Don't use `@view` b/c not performant on GPU's
    # get the current timelevel of prognostic variables
    normalVelocity = @view Prog.normalVelocity[:,:,end]
    layerThickness = @view Prog.layerThickness[:,:,end]

    # initialize the kernel
    kernel! = explicit_bottom_drag_tendency_kernel!(backend)
    # use kernel to compute explicit bottom drg
    kernel!(tendNormalVelocity,
            kineticEnergyCell,
            normalVelocity,
            layerThickness,
            cellsOnEdge,
            maxLevelEdge.Top, 
            dragCoefficient,
            nEdges,
            ndrange = nEdges)
    # sync the backend 
    KA.synchronize(backend)

    # pack the tendecy pack into the struct for further computation
    @pack! Tend = tendNormalVelocity
end


@kernel function explicit_bottom_drag_tendency_kernel!(tend,
                                                       @Const(KECell),
                                                       @Const(normalVelocity),
                                                       @Const(layerThickness),
                                                       @Const(cellsOnEdge), 
                                                       @Const(maxLevelEdgeTop), 
                                                       dragCoefficient,
                                                       nEdges)

    # global indices over nEdges
    iEdge = @index(Global, Linear)
    
    if iEdge < nEdges + 1
        # 
        @inbounds @private k = maxLevelEdgeTop[iEdge]
        #
        @inbounds @private jCell1 = cellsOnEdge[1,iEdge]      
        @inbounds @private jCell2 = cellsOnEdge[2,iEdge]

        if k > 0
            tend[k, iEdge] -= dragCoefficient * normalVelocity[k, iEdge] *
                              sqrt(KECell[k, jCell1] + KECell[k, jCell2]) /
                              layerThickness[k, iEdge]
        end
    end
end
