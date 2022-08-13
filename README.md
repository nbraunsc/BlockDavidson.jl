# BlockDavidson

[![Build Status](https://github.com/nmayhall-vt/BlockDavidson.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/nmayhall-vt/BlockDavidson.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/nmayhall-vt/BlockDavidson.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/nmayhall-vt/BlockDavidson.jl)

---
Simple code to solve for the eigenvectors/eigenvalues of a matrix or `LinearMap`. Preconditioner has not yet been implemented, so this is currently just a simple block lanczos code. 

## Example usage 
### Explicit matrix diagonalization
```julia
dav = Davidson(A)
e,v = solve(dav)
```
### Matrix-free diagonalization 
We can also diagonalize an implicit matrix by defining a function `matvec` that algorithmically implments the action of `A` onto a vector or set of vectors. We can also specify several settings, including providing an initial guess `v_guess`:
```julia
using LinearMaps
function matvec(v)
    return A*v. # implement this however is appropriate
end


lmap = LinearMap(matvec)
dav = Davidson(lmap; max_iter=200, max_ss_vecs=8, tol=1e-6, nroots=6, v0=v_guess, lindep_thresh=1e-10)
e,v = solve(dav)
```
