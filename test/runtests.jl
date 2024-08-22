using Test

include("utilities.jl")

@testset "Moka" begin 

    @testset "Infrastructre Test" begin
        include("infra/test_Config.jl")
        include("infra/test_timeManager.jl")
    end

    @testset "Operator/Numerical Tests" begin 
        include("ocn/test_Operators.jl")
        include("ocn/test_Diagnostics.jl")
    end
    
    @testset "Enzyme Tests" begin 
        include("enzyme/test_Enzyme_Operators.jl")
    end
end
