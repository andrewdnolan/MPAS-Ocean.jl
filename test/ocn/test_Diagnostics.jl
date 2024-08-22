using Test
using CUDA
using MOKA
using UnPack
using LinearAlgebra
using CUDA: @allowscalar

import Adapt
import Downloads
import KernelAbstractions as KA

# include the testcase definition utilities
include("../utilities.jl")

mesh_fn = DownloadMesh(PlanarTest)
backend = KA.CPU()

# Read in the purely horizontal doubly periodic testing mesh
HorzMesh = ReadHorzMesh(mesh_fn; backend=backend)
# Create a dummy vertical mesh from the horizontal mesh
VertMesh = VerticalMesh(HorzMesh; nVertLevels=10, backend=backend)
# Create a the full Mesh strucutre 
MPASMesh = Mesh(HorzMesh, VertMesh)

# get some dimension information
nEdges = HorzMesh.Edges.nEdges
nCells = HorzMesh.PrimaryCells.nCells
nVertices = HorzMesh.DualCells.nVertices
nVertLevels = VertMesh.nVertLevels

setup = TestSetup(MPASMesh, PlanarTest; backend=backend)


velocityX(args...) = 𝐅ˣ(args...)
velocityY(args...) = 𝐅ʸ(args...)

𝐅ˣ(::Type{Cell}, test::TestSetup, args...) = 𝐅ˣ(test.xᶜ, test.yᶜ, test, args...)
𝐅ʸ(::Type{Cell}, test::TestSetup, args...) = 𝐅ʸ(test.xᶜ, test.yᶜ, test, args...)

𝐅ˣ(::Type{Edge}, test::TestSetup, args...) = 𝐅ˣ(test.xᵉ, test.yᵉ, test, args...)
𝐅ʸ(::Type{Edge}, test::TestSetup, args...) = 𝐅ʸ(test.xᵉ, test.yᵉ, test, args...)

function 𝐅ˣ(x, y, test::TestSetup, ::Type{PlanarTest})
    @unpack Lx, Ly = test 

    return @. sin(2.0 * pi * x / Lx) * cos(2.0 * pi * y / Ly)
end

function 𝐅ʸ(x, y, test::TestSetup, ::Type{PlanarTest})
    @unpack Lx, Ly = test 

    return @. cos(2.0 * pi * x / Lx) * sin(2.0 * pi * y / Ly)
end

function kineticEnergy(loc, setup, type)
    @unpack nVertLevels = setup

    KE = (velocityX(loc, setup, type) .* velocityX(loc, setup, type) .+ 
          velocityY(loc, setup, type) .* velocityY(loc, setup, type)) ./ 2.

    # return nVertLevels time tiled version of the array
    return repeat(KE', outer=[nVertLevels, 1])
end

###
### Kinetic Energy Test
###

# Scalar field defined at cell edges
normalVelocityEdge = 𝐅ₑ(setup, PlanarTest)
# scratch array at cell edges used in kineticEnergy calculation
scratchEdge = KA.zeros(backend, Float64, (nVertLevels, nEdges))

# Exact (K)inetic (E)nergy at cell centers
KEAnn = kineticEnergy(Cell, setup, PlanarTest)
# Numerical Kinetic Energy using KernelAbstractions operator
KENum = KA.zeros(backend, Float64, (nVertLevels, nCells))
@allowscalar MOKA.calculate_kineticEnergyCell!(KENum,
                                               scratchEdge,
                                               normalVelocityEdge,
                                               MPASMesh)

# Asses the numerical error of discrete Kinetic Energy calculation
KEError = ErrorMeasures(KENum, KEAnn, HorzMesh, Cell)

@test KEError.L_inf ≈ 0.00994439065100057897 atol=atol
@test KEError.L_two ≈ 0.00703403756741667954 atol=atol
