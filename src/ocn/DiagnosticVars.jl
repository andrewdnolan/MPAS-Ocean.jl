import Adapt

using CUDA
using KernelAbstractions

mutable struct DiagnosticVars{F <: AbstractFloat, FV2 <: AbstractArray{F,2}}
    
    # var: layer thickness averaged from cell centers to edges [m]
    # dim: (nVertLevels, nEdges)
    layerThicknessEdge::FV2
    
    # var: ....
    # vim: (nVertLevels, nEdges)
    thicknessFlux::FV2

    # var: divergence of horizonal velocity [s^{-1}]
    # dim: (nVertLevels, nCells)
    velocityDivCell::FV2

    # var: curl of horizontal velocity [s^{-1}]
    # dim: (nVertLevels, nVertices)
    relativeVorticity::FV2

    # var: kinetic energy of horizonal velocity on cells [m^{2} s^{-2}]
    # dim: (nVertLevels, nCells)
    kineticEnergyCell::FV2

    #= Performance Note: 
    # ###########################################################
    #  While these can be stored as diagnostic variales I don't 
    #  really think we need to do that. Only used locally within 
    #  tendency calculations, so should be more preformant to 
    #  calculate the values locally within the tendency loops. 
    # ###########################################################
     
    # var: flux divergence [m s^{-1}] ? 
    # dim: (nVertLevels, nCells)
    div_hu::Array{F,2}
    
    # var: Gradient of sea surface height at edges. [-] 
    # dim: (nEdges), Time)?
    gradSSH::Array{F,1}
    =#
    
    #= UNUSED FOR NOW:
    # var: horizontal velocity, tangential to an edge [m s^{-1}] 
    # dim: (nVertLevels, nEdges)
    tangentialVelocity::Array{F, 2}

    =# 

    function DiagnosticVars(layerThicknessEdge::AT2D, 
                            thicknessFlux::AT2D, 
                            velocityDivCell::AT2D, 
                            relativeVorticity::AT2D, 
                            kineticEnergyCell::AT2D) where {AT2D}
        # pack all the arguments into a tuple for type and backend checking
        args = (layerThicknessEdge, thicknessFlux,
                velocityDivCell, relativeVorticity, kineticEnergyCell)
        
        # check the type names; irrespective of type parameters
        # (e.g. `Array` instead of `Array{Float64, 1}`)
        check_typeof_args(args)
        # check that all args are on the same backend
        check_args_backend(args)
        # check that all args have the same `eltype` and get that type
        type = check_eltype_args(args)

        new{type, AT2D}(layerThicknessEdge,
                        thicknessFlux,
                        velocityDivCell,
                        relativeVorticity, 
                        kineticEnergyCell)
    end
end 
 
function DiagnosticVars(config::GlobalConfig, Mesh::Mesh; backend=KA.CPU())

    @unpack HorzMesh, VertMesh = Mesh    
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    nEdges = Edges.nEdges
    nCells = PrimaryCells.nCells
    nVertices = DualCells.nVertices
    nVertLevels = VertMesh.nVertLevels
    
    # Here in the init function is where some sifting through will 
    # need to be done, such that only diagnostic variables required by 
    # the `Config` or requested by the `streams` will be activated. 
    
    # create zero vectors to store diagnostic variables, on desired backend
    thicknessFlux      = KA.zeros(backend, Float64, nVertLevels, nEdges) 
    velocityDivCell    = KA.zeros(backend, Float64, nVertLevels, nCells)
    kineticEnergyCell  = KA.zeros(backend, Float64, nVertLevels, nCells)
    relativeVorticity  = KA.zeros(backend, Float64, nVertLevels, nVertices)
    layerThicknessEdge = KA.zeros(backend, Float64, nVertLevels, nEdges) 

    DiagnosticVars(layerThicknessEdge,
                   thicknessFlux,
                   velocityDivCell,
                   relativeVorticity, 
                   kineticEnergyCell)
end 

function Adapt.adapt_structure(to, x::DiagnosticVars)
    return DiagnosticVars(Adapt.adapt(to, x.layerThicknessEdge),
                          Adapt.adapt(to, x.thicknessFlux), 
                          Adapt.adapt(to, x.velocityDivCell),
                          Adapt.adapt(to, x.relativeVorticity),
                          Adapt.adapt(to, x.kineticEnergyCell))
end

function diagnostic_compute!(Mesh::Mesh,
                             Diag::DiagnosticVars,
                             Prog::PrognosticVars;
                             backend = KA.CPU())

    calculate_thicknessFlux!(Diag, Prog, Mesh; backend = backend)
    calculate_velocityDivCell!(Diag, Prog, Mesh; backend = backend)
    calculate_relativeVorticity!(Diag, Prog, Mesh; backend = backend)
    calculate_kineticEnergyCell!(Diag, Prog, Mesh; backend = backend)
    calculate_layerThicknessEdge!(Diag, Prog, Mesh; backend = backend)
end 

function calculate_layerThicknessEdge!(Diag::DiagnosticVars,
                                       Prog::PrognosticVars,
                                       Mesh::Mesh;
                                       backend = KA.CPU())
    
    #layerThickness = Prog.layerThickness[:,:,end]
    @unpack layerThicknessEdge = Diag 
    
    interpolateCell2Edge!(layerThicknessEdge, 
                          Prog.layerThickness[end],
                          Mesh; backend = backend)

    @pack! Diag = layerThicknessEdge
end 

function calculate_thicknessFlux!(Diag::DiagnosticVars,
                                  Prog::PrognosticVars,
                                  Mesh::Mesh;
                                  backend = CUDABackend())

    normalVelocity = Prog.normalVelocity[end]
    @unpack thicknessFlux, layerThicknessEdge = Diag 

    nthreads = 100
    kernel!  = compute_thicknessFlux!(backend, nthreads)

    kernel!(thicknessFlux, Prog.normalVelocity[end], layerThicknessEdge, size(normalVelocity)[2], ndrange=size(normalVelocity)[2])
    #kernel!(thicknessFlux, Prog.normalVelocity, layerThicknessEdge, ndrange=(size(Prog.normalVelocity)[1],size(Prog.normalVelocity)[2]))

    @pack! Diag = thicknessFlux
end

@kernel function compute_thicknessFlux!(thicknessFlux,
                                        @Const(normalVelocity),
                                        @Const(layerThicknessEdge),
                                        arrayLength)

    j = @index(Global, Linear)
    if j < arrayLength + 1
        @inbounds thicknessFlux[1,j] = normalVelocity[1,j] * layerThicknessEdge[1,j]
    end

    #k, j = @index(Global, NTuple)
    #if j < arrayLength + 1
    #    @inbounds thicknessFlux[k,j] = normalVelocity[k,j,end] * layerThicknessEdge[k,j]
    #end
    @synchronize()
end

function calculate_velocityDivCell!(Diag::DiagnosticVars, 
                                    Prog::PrognosticVars, 
                                    Mesh::Mesh;
                                    backend = KA.CPU()) 
    
    normalVelocity = Prog.normalVelocity[end]

    # I think the issue is that this doesn't create a new array while the old version does... we need a
    # new array for temporary data

    # layerThicknessEdge is used here to temporarily store intermdeiate results. It will be reset when it is acually
    # used as a diagnostic variable
    @unpack velocityDivCell, layerThicknessEdge = Diag


    DivergenceOnCell!(velocityDivCell, normalVelocity, layerThicknessEdge, Mesh; backend=backend)

    @pack! Diag = velocityDivCell
end

function calculate_relativeVorticity!(Diag::DiagnosticVars, 
                                      Prog::PrognosticVars, 
                                      Mesh::Mesh;
                                      backend = KA.CPU()) 

    #normalVelocity = Prog.normalVelocity[:,:,end]

    @unpack relativeVorticity = Diag

    CurlOnVertex!(relativeVorticity, Prog.normalVelocity[end], Mesh; backend=backend)

    @pack! Diag = relativeVorticity
end


function calculate_kineticEnergyCell!(Diag::DiagnosticVars, 
                                      Prog::PrognosticVars, 
                                      Mesh::Mesh;
                                      backend = KA.CPU())

    # let's add scratch array to diag struct, but we'll need the zero out
    # function from joes branch
    normalVelocity = Prog.normalVelocity[:,:,end]
    scratchEdge    = deepcopy(normalVelocity)
    
    @unpack kineticEnergyCell = Diag

    calculate_kineticEnergyCell(kineticEnergyCell,
                                scratchEdge,           
                                VecEdge,
                                Mesh::Mesh;
                                backend = KA.CPU())

    @pack! Diag = kineticEnergyCell 
end

# probably should check the size/backend/eltypes of array args here
function calculate_kineticEnergyCell!(kineticEnergyCell,
                                      scratchEdge,
                                      VecEdge,
                                      Mesh::Mesh;
                                      backend = KA.CPU())
    
    @unpack HorzMesh, VertMesh = Mesh    
    @unpack PrimaryCells, DualCells, Edges = HorzMesh
    
    @unpack nVertLevels = VertMesh 
    @unpack dvEdge, dcEdge, nEdges = Edges
    @unpack nCells, nEdgesOnCell, edgesOnCell, areaCell = PrimaryCells
    
    kernel1! = KineticEnergyCell_P1!(backend)
    kernel2! = KineticEnergyCell_P2!(backend)

    kernel1!(scratchEdge,
             VecEdge,
             dvEdge,
             dcEdge,
             workgroupsize=64,
             ndrange=(nEdges, nVertLevels))

    kernel2!(kineticEnergyCell,
             scratchEdge,
             nEdgesOnCell,
             edgesOnCell,
             areaCell,
             workgroupsize=32,
             ndrange=(nCells, nVertLevels))

    KA.synchronize(backend)
end

@kernel function KineticEnergyCell_P1!(tmp,
                                       @Const(VecEdge),
                                       @Const(dvEdge), 
                                       @Const(dcEdge))

    iEdge, k = @index(Global, NTuple)
    @inbounds tmp[k,iEdge] = VecEdge[k,iEdge] * VecEdge[k,iEdge] *
                             0.5 * dvEdge[iEdge] * dcEdge[iEdge] # AreaEdge
    @synchronize()
end

@kernel function KineticEnergyCell_P2!(KineticEnergyCell,
                                       @Const(tmp),
                                       @Const(nEdgesOnCell),
                                       @Const(edgesOnCell),
                                       @Const(areaCell))
    iCell, k = @index(Global, NTuple)

    KineticEnergyCell[k,iCell] = 0.0

    # loop over number of edges in primary cell
    for i in 1:nEdgesOnCell[iCell]
        @inbounds iEdge = edgesOnCell[i,iCell]
        @inbounds KineticEnergyCell[k,iCell] += tmp[k,iEdge] #* 0.5
    end

    KineticEnergyCell[k,iCell] /= areaCell[iCell] * 2.0
    @synchronize()
end
