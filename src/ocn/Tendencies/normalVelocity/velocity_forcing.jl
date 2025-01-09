function velocity_forcing_tendency!(Tend::TendencyVars, 
                                    Prog::PrognosticVars,
                                    Diag::DiagnosticVars, 
                                    Mesh::Mesh, 
                                    Forcing::ForcingVars;
                                    backend = KA.CPU())

    # if surface forcing is present, the compute tendency
    if !isnothing(Forcing.sfcStress)
        surface_stress_forcing_tendency!(
            Tend, Prog, Diag, Mesh, Forcing; backend=backend
        )
    end
end

function surface_stress_forcing_tendency!(Tend::TendencyVars, 
                                          Prog::PrognosticVars,
                                          Diag::DiagnosticVars, 
                                          Mesh::Mesh, 
                                          Forcing::ForcingVars;
                                          backend = KA.CPU())

    @unpack HorzMesh, VertMesh = Mesh    
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    @unpack maxLevelEdge = VertMesh 
    @unpack nEdges, edgeMask = Edges
   
    @unpack sfcStress = Forcing
    @unpack layerThicknessEdge = Diag

    # unpack the normal velocity tendency term
    @unpack tendNormalVelocity = Tend 
    
    # initialize the kernel
    nthreads = 50
    kernel! = surfaceStressForcing!(backend, nthreads)
    # use kernel to compute gradient
    kernel!(tendNormalVelocity,
            sfcStress,
            layerThicknessEdge,
            maxLevelEdge.Top,
            edgeMask,
            ndrange=nEdges)
    # sync the backend 
    KA.synchronize(backend)
    
    # pack the tendecy pack into the struct for further computation
    @pack! Tend = tendNormalVelocity 
end

@kernel function surfaceStressForcing!(tendency,
                                       @Const(sfcStress),
                                       @Const(layerThicknessEdge),
                                       @Const(maxLevelEdgeTop),
                                       @Const(edgeMask))

    # global indices over nEdges
    iEdge = @index(Global, Linear)

    for k in 1:maxLevelEdgeTop[iEdge]
        # gradient on edges calculation 
        tendency[k, iEdge] -= edgeMask[k, iEdge] * sfcStress[iEdge] /\
                              (layeThicknessEdge[k, iEdge] * 1000.0)
    end
end
