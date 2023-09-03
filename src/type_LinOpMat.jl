"""
    matvec
    dim::Int
    sym::Bool
"""
mutable struct LinOpMat{T} <: AbstractMatrix{T} 
    matvec
    dim::Int
    sym::Bool
end

Base.size(lop::LinOpMat{T}) where {T} = return (lop.dim,lop.dim)
Base.:(*)(lop::LinOpMat{T}, v::AbstractVector{T}) where {T} = return lop.matvec(v)
Base.:(*)(lop::LinOpMat{T}, v::AbstractMatrix{T}) where {T} = return lop.matvec(v)
issymmetric(lop::LinOpMat{T}) where {T} = return lop.sym