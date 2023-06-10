using BlockDavidson
using LinearAlgebra
using Random

Random.seed!(2)

N = 10_000
A = -10 .* diagm(rand(N)) + .001 * rand(N,N) 
A += A'

v0 = qr(rand(N,2)).Q[:,1:3];