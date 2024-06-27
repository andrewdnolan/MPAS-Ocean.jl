using Dates
using CUDA
using MOKA
using Statistics
using KernelAbstractions

using Enzyme

# Might need to remove these:
#Enzyme.EnzymeRules.inactive_type(::Type{<:MOKA.ModelSetup}) = true
#Enzyme.EnzymeRules.inactive_type(::Type{<:MOKA.Clock}) = true

const KA=KernelAbstractions

# file path to the config file. Should be parsed from the command line 
#config_fp = "../../TestData/inertial_gravity_wave_100km.yml"
#config_fp = "/global/homes/a/anolan/MPAS-Ocean.jl/bare_minimum.yml"
config_fp = "./config.yml"

function ocn_run(config_fp)

    #
    # Setup for model
    #
    
    backend = KA.CPU()
    #backend = CUDABackend()

    # Initialize the Model  
    Setup, Diag, Tend, Prog = ocn_init(config_fp, backend = backend)
    
    mesh = Setup.mesh 
    config = Setup.config
    
    # this is hardcoded for now, but should really be set accordingly in the 
    # yaml file
    #dt = floor(3.0 * mean(mesh.dcEdge) / 1e3)
    dcEdge = mesh.HorzMesh.Edges.dcEdge
    dt = floor(2 * (mean(dcEdge) / 1e3) * mean(dcEdge) / 200e3) 
    changeTimeStep!(Setup.timeManager, Second(dt))
    
    clock = Setup.timeManager
    
    simulationAlarm = clock.alarms["simulation_end"]
    outputAlarm = clock.alarms["outputAlarm"]

    #
    # Actual Model Run with AD
    #

    d_Prog = Enzyme.make_zero(Prog)
    d_Diag = Enzyme.make_zero(Diag)
    d_Tend = Enzyme.make_zero(Tend)
    d_Setup = Enzyme.make_zero(Setup)
    d_ForwardEuler = Enzyme.make_zero(ForwardEuler)
    d_clock = Enzyme.make_zero(clock)
    d_simulationAlarm = Enzyme.make_zero(simulationAlarm)
    d_outputAlarm = Enzyme.make_zero(outputAlarm)

    #@show d_Setup.mesh.VertMesh.restingThicknessSum

    # Let's see how to increase the variation in ocean ssh:
    #=
    for j = 1:size(Prog.ssh)[end]
        if Prog.ssh[j] > 0
            d_Prog.ssh[j] = 1.0
        elseif Prog.ssh[j] < 0
            d_Prog.ssh[j] = -1.0
        end
    end
    =#
    #d_Prog.layerThickness[1,1,1] = 1.0

    #old_Prog = deepcopy(Prog)
    #=
    d_sum = autodiff(Enzyme.Reverse,
             ocn_run_loop,
             Duplicated(Prog, d_Prog),
             Duplicated(Diag, d_Diag),
             Duplicated(Tend, d_Tend),
             Duplicated(Setup, d_Setup),
             Duplicated(ForwardEuler, d_ForwardEuler),
             Duplicated(clock, d_clock),
             Duplicated(simulationAlarm, d_simulationAlarm),
             Duplicated(outputAlarm, d_outputAlarm),
             )
    =#
    autodiff(Enzyme.Reverse,
             ocn_timestep_ForwardEuler,
             Duplicated(Prog, d_Prog),
             Duplicated(Diag, d_Diag),
             Duplicated(Tend, d_Tend),
             Duplicated(Setup, d_Setup))
    
    #ocn_timestep_ForwardEuler(Prog, Diag, Tend, Setup; backend=backend)
    #=
    @show Prog.ssh - old_Prog.ssh
    @show Prog.layerThickness - old_Prog.layerThickness
    @show Prog.normalVelocity - old_Prog.normalVelocity
    =#

    #ocn_run_loop(Prog, Diag, Tend, Setup, ForwardEuler, clock, simulationAlarm, outputAlarm; backend=backend)

    #
    # Writing to outputs
    #
    
    # Only suport i/o at the end of the simulation for now 
    write_netcdf(Setup, Diag, Prog, d_Prog)
    
    backend = get_backend(Tend.tendNormalVelocity)
    arch = typeof(backend) <: KA.GPU ? "GPU" : "CPU" 

    println("Moka.jl ran on $arch")
    println(clock.currTime)
end

function ocn_run_loop(Prog, Diag, Tend, Setup, ForwardEuler, clock, simulationAlarm, outputAlarm; backend=KA.CPU())
    #global i = 0
    # Run the model 
    #while !isRinging(simulationAlarm)
    for i = 1:90
    
        advance!(clock)
    
        #global i += 1
    
        ocn_timestep_ForwardEuler(Prog, Diag, Tend, Setup; backend=backend)
        
        if isRinging(outputAlarm)
            # should be doing i/o in here, using a i/o struct
            reset!(outputAlarm)
        end
    end

    sum = 0.0
    ssh_length = size(Prog.ssh)[1]
    for j = 1:ssh_length
        sum = sum + Prog.ssh[j]^2
    end

    @show sum

    return sum
end

if abspath(PROGRAM_FILE) == @__FILE__
    if isfile(ARGS[1])
        ocn_run(ARGS[1])
    else 
        error("yaml config file invalid")
    end
end
