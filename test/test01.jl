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
    e,v = eigs(dav)
    e_ref = -18.260037795157675
    @test isapprox(e[1], -18.260037795157675)

    lmap = LinearMap(A)

    dav = Davidson(lmap)
    e,v = eigs(dav)
    e_ref = -18.260037795157675
    @test isapprox(e[1], -18.260037795157675)

    e_ref = [
             -18.260037795157675
             -17.9716411644818
             -17.47598854674256
             -17.184105784827796
             -16.939563610543257
             -16.83937885452674
            ]
    # now with more settings specified and roots
    dav = Davidson(lmap; max_iter=200, max_ss_vecs=8, tol=1e-6, nroots=6)
    e,v = eigs(dav)
    @test all(isapprox.(e, e_ref, atol=1e-10))

    display(v'*Matrix(lmap*v))
    # now test with initial guess
    e,v = eigs(Davidson(lmap; max_iter=2, max_ss_vecs=8, tol=1e-8, nroots=6, v0=v))
    @test all(isapprox.(e, e_ref, atol=1e-10))
end
