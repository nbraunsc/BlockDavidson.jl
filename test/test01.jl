using BlockDavidson
using Random
using LinearAlgebra
using Test
using LinearMaps

@testset "BlockDavidson" begin
    Random.seed!(2)

    A = rand(500,500) .- .5;
    A += A';

    #F = eigen(A);
    #display(F.values)

    dav = Davidson(A)
    e,v = solve(dav)
    e_ref = -18.260037795157675
    @test isapprox(e[1], -18.260037795157675)

    lmap = LinearMap(A)

    dav = Davidson(lmap)
    e,v = solve(dav)
    e_ref = -18.260037795157675
    @test isapprox(e[1], -18.260037795157675)

end
