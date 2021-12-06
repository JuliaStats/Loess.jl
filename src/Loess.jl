module Loess

import Distances.euclidean

using Statistics, LinearAlgebra

export loess, predict

include("kd.jl")


struct LoessModel{B, T <: AbstractFloat, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
    xs::M                       # An n by m predictor matrix containing n observations from m predictors
    ys::V                       # A length n response vector
    bs::Matrix{T}               # Least squares coefficients
    verts::Dict{Vector{T}, Int} # kd-tree vertexes mapped to indexes
    kdtree::KDTree{T,M}
end

"""
    loess(xs, ys, normalize=true, span=0.75, degree=2)

Fit a loess model.

Args:
  - `xs`: A `n` by `m` matrix with `n` observations from `m` independent predictors
  - `ys`: A length `n` response vector.
  - `normalize`: Normalize the scale of each predicitor. (default true when `m > 1`)
  - `span`: The degree of smoothing, typically in [0,1]. Smaller values result in smaller
      local context in fitting.
  - `degree`: Polynomial degree.

Returns:
  A fit `LoessModel`.

"""
function loess(xs::M, ys::V;
               normalize::Bool=true,
               span::AbstractFloat=0.75,
               degree::Integer=2) where {T <: AbstractFloat, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
    if size(xs, 1) != size(ys, 1)
        throw(DimensionMismatch("Predictor and response arrays must of the same length"))
    end

    n, m = size(xs)
    q = ceil(Int, (span * n))
    if q < degree + 1
        throw(ArgumentError("neighborhood size must be larger than degree+1=$(degree + 1) but was $q. Try increasing the value of span."))
    end

    # TODO: We need to keep track of how we are normalizing so we can
    # correctly apply predict to unnormalized data. We should have a normalize
    # function that just returns a vector of scaling factors.
    if normalize && m > 1
        xs = tnormalize!(copy(xs))
    end

    kdtree = KDTree(xs, 0.05 * span)

    # map verticies to their index in the bs coefficient matrix
    verts = Dict{Vector{T}, Int}()
    for (k, vert) in enumerate(kdtree.verts)
        verts[vert] = k
    end

    # Fit each vertex
    ds = Array{T}(undef, n) # distances
    perm = collect(1:n)
    bs = Array{T}(undef, length(kdtree.verts), 1 + degree * m)

    # TODO: higher degree fitting
    us = Array{T}(undef, q, 1 + degree * m)
    vs = Array{T}(undef, q)

    for (vert, k) in verts
        # reset perm
        for i in 1:n
            perm[i] = i
        end

        # distance to each point
        for i in 1:n
            ds[i] = euclidean(vec(vert), vec(xs[i,:]))
        end

        # copy the q nearest points to vert into X
        partialsort!(perm, q, by=i -> ds[i])
        dmax = maximum([ds[perm[i]] for i = 1:q])

        for i in 1:q
            pᵢ = perm[i]
            w = tricubic(ds[pᵢ] / dmax)
            us[i,1] = w
            for j in 1:m
                x = xs[pᵢ, j]
                wxl = w
                for l in 1:degree
                    wxl *= x
                    us[i, 1 + (j - 1)*degree + l] = wxl # w*x^l
                end
            end
            vs[i] = ys[pᵢ] * w
        end

        if VERSION < v"1.7.0-DEV.1188"
            F = qr(us, Val(true))
        else
            F = qr(us, ColumnNorm())
        end
        bs[k,:] = F\vs
    end

    LoessModel{(m>1),T,V,M}(xs, ys, bs, verts, kdtree)
end

loess(xs::AbstractVector{T}, ys::AbstractVector{T}; kwargs...) where {T<:AbstractFloat} =
    loess(reshape(xs, (length(xs), 1)), ys; kwargs...)

function loess(xs::AbstractArray{T,N}, ys::AbstractVector{S}; kwargs...) where {T,N,S}
    R = float(promote_type(T, S))
    loess(convert(AbstractArray{R,N}, xs), convert(AbstractVector{R}, ys); kwargs...)
end


# Predict response values from a trained loess model and predictor observations.
#
# Loess (or at least most implementations, including this one), does not perform,
# extrapolation, so none of the predictor observations can exceed the range of
# values in the data used for training.
#
# Args:
#   zs/z: An n' by m matrix of n' observations from m predictors. Where m is the
#       same as the data used in training.
#
# Returns:
#   A length n' vector of predicted response values.
#

# univariate model, prediction at single point
function predict(model::LoessModel{false,T,V,M}, z::T) where {T <: AbstractFloat, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
    adjacent_verts = traverse(model.kdtree, T[z])
    @assert(length(adjacent_verts) == 2)
    v₁, v₂ = adjacent_verts[1][1], adjacent_verts[2][1]
    interpolate(model,z,v₁,v₂)
end

# univariate model, prediction at multiple points
function predict(model::LoessModel{false,T,V,M}, zs::AbstractVector{T}) where {T <: AbstractFloat, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
    map(z->predict(model, z),zs)
end

# multivariate model, prediction at single point
function predict(model::LoessModel{true,T,V,M}, z::AbstractVector{T}) where {T <: AbstractFloat, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
    if length(z) != size(model.xs, 2)
        throw(DimensionMismatch("$(size(model.xs, 2))-dimensional model applied to length $(length(z)) vector"))
    end
    error("Multivariate blending not yet implemented")
    # TODO:
    #   1. Univariate linear interpolation between adjacent verticies.
    #   2. Blend these estimates. (I'm not sure how this is done.)
end

# multivariate model, prediction at multiple points
function predict(model::LoessModel{true,T,V,M}, zs::AbstractMatrix{T}) where {T <: AbstractFloat, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
    map(z->predict(model, z), eachrow(zs))
end

"""Univariate interpolation"""
function interpolate(model, z::T,v₁::T, v₂::T) where {T <: AbstractFloat}
    zs=T[z]
    if z == v₁ || z == v₂
        return evalpoly(zs, model.bs[model.verts[[z]],:])
    end

    u = (z - v₁)/(v₂ - v₁)

    y1 = evalpoly(zs, model.bs[model.verts[[v₁]],:])
    y2 = evalpoly(zs, model.bs[model.verts[[v₂]],:])
    return (one(u) - u) * y1 + u * y2
end

"""
    tricubic(u)

Tricubic weight function.

Args:
  - `u`: Distance between 0 and 1

Returns:
  A weighting of the distance `u`

"""
tricubic(u) = (1 - u^3)^3


"""
    evalpoly(xs,bs)

Evaluate a multivariate polynomial with coefficients `bs` at `xs`.  `bs` should be of length
`1+length(xs)*d` where `d` is the degree of the polynomial.

    bs[1] + xs[1]*bs[2] + xs[1]^2*bs[3] + ... + xs[end]^d*bs[end]

"""
function evalpoly(xs, bs)
    m = length(xs)
    degree = div(length(bs) - 1, m)
    y = bs[1]
    for i in 1:m
        x = xs[i]
        xx = x
        y += xx * bs[1 + (i-1)*degree + 1]
        for l in 2:degree
            xx *= x
            y += xx * bs[1 + (i-1)*degree + l]
        end
    end
    y
end

"""
    tnormalize!(x,q)

Default normalization procedure for predictors.

This simply normalizes by the mean of everything between the 10th an 90th percentiles.

Args:
  - `xs`: a matrix of predictors
  - `q`: cut the ends of at quantiles `q` and `1-q`

Modifies:
  `xs`
"""
function tnormalize!(xs::AbstractMatrix{T}, q::T=0.1) where T <: AbstractFloat
    n, m = size(xs)
    cut = ceil(Int, (q * n))
    for j in 1:m
        tmp = sort!(xs[:,j])
        xs[:,j] ./= mean(tmp[cut+1:n-cut])
    end
    xs
end


end
