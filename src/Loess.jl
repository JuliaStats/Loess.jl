module Loess

import Distances: euclidean
import StatsAPI: fitted, modelmatrix, predict, residuals, response

using Statistics, LinearAlgebra

export loess, fitted, modelmatrix, predict, residuals, response

include("kd.jl")


struct LoessModel{T <: AbstractFloat}
    xs::Matrix{T} # An n by m predictor matrix containing n observations from m predictors
    ys::Vector{T} # A length n response vector
    predictions_and_gradients::Dict{Vector{T}, Vector{T}} # kd-tree vertexes mapped to prediction and gradient at each vertex
    kdtree::KDTree{T}
end

modelmatrix(model::LoessModel) = model.xs

response(model::LoessModel) = model.ys

function _loess(
    xs::AbstractMatrix{T},
    ys::AbstractVector{T};
    normalize::Bool = true,
    span::AbstractFloat = 0.75,
    degree::Integer = 2,
    cell::AbstractFloat = 0.2
) where T<:AbstractFloat
    Base.require_one_based_indexing(xs)
    Base.require_one_based_indexing(ys)

    if size(xs, 1) != length(ys)
        throw(DimensionMismatch("Predictor and response arrays must of the same length"))
    end
    if isempty(ys)
        throw(ArgumentError("input arrays are empty"))
    end

    n, m = size(xs)
    q = max(1, floor(Int, span * n))

    # TODO: We need to keep track of how we are normalizing so we can
    # correctly apply predict to unnormalized data. We should have a normalize
    # function that just returns a vector of scaling factors.
    if normalize && m > 1
        throw(ArgumentError("higher dimensional models not yet supported"))
        xs = tnormalize!(copy(xs))
    end

    kdtree = KDTree(xs, cell * span, 0)

    # map verticies to their prediction and prediction gradient
    predictions_and_gradients = Dict{Vector{T}, Vector{T}}()

    # Fit each vertex
    ds = Array{T}(undef, n) # distances
    perm = collect(1:n)

    # Initialize the regression arrays
    us = Array{T}(undef, q, 1 + degree * m)
    du1dt = zeros(T, m, 1 + degree * m)
    vs = Array{T}(undef, q)

    for vert in kdtree.verts
        # reset perm
        for i in 1:n
            perm[i] = i
        end

        # distance to each point
        @inbounds for i in 1:n
            s = zero(T)
            for j in 1:m
                s += (xs[i, j] - vert[j])^2
            end
            ds[i] = sqrt(s)
        end

        # find the q closest points
        partialsort!(perm, 1:q, by=i -> ds[i])
        dmax = maximum([ds[perm[i]] for i = 1:q])
        dmax = iszero(dmax) ? one(dmax) : dmax

        for i in 1:q
            pᵢ = perm[i]
            w = sqrt(tricubic(ds[pᵢ] / dmax))
            us[i, 1] = w
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

        # Compute the gradient of the vertex
        pᵢ = perm[1]
        for j in 1:m
            x = xs[pᵢ, j]
            xl = one(x)
            for l in 1:degree
                du1dt[j, 1 + (j - 1)*degree + l] = l * xl
                xl *= x
            end
        end
        
        coefs = Matrix{Real}(undef, size(us, 2), 1)
        try
            coefs = cholesky(us' * us) \ (us' * vs)
        catch PosDefException
            if VERSION < v"1.7.0-DEV.1188"
                F = qr(us, Val(true))
            else
                F = qr(us, ColumnNorm())
            end
            coefs = F \ vs
        end

        predictions_and_gradients[vert] = [
            us[1, :]' * coefs; # the prediction
            du1dt * coefs      # the gradient of the prediction
        ]
    end

    LoessModel(convert(Matrix{T}, xs), convert(Vector{T}, ys), predictions_and_gradients, kdtree)
end

"""
    loess(xs, ys; normalize=true, span=0.75, degree=2)

Fit a loess model.

Args:
  - `xs`: A `n` by `m` matrix with `n` observations from `m` independent predictors
  - `ys`: A length `n` response vector.
  - `normalize`: Normalize the scale of each predicitor. (default true when `m > 1`)
  - `span`: The degree of smoothing, typically in [0,1]. Smaller values result in smaller
      local context in fitting.
  - `degree`: Polynomial degree.
  - `cell`: Control parameter for bucket size. Internal interpolation nodes will be
    added to the K-D tree until the number of bucket element is below `n * cell * span`.

Returns:
  A fit `LoessModel`.

"""
function loess(
    xs::AbstractMatrix{T},
    ys::AbstractVector{T};
    normalize::Bool = true,
    span::AbstractFloat = 0.75,
    degree::Integer = 2,
    cell::AbstractFloat = 0.2
) where T<:AbstractFloat
    _loess(xs, ys; normalize, span, degree, cell)
end

loess(xs::AbstractVector{T}, ys::AbstractVector{S}; kwargs...) where {T,S} =
    loess(reshape(xs, (length(xs), 1)), ys; kwargs...)

function loess(xs::AbstractMatrix{T}, ys::AbstractVector{S}; kwargs...) where {T,S}
    R = float(promote_type(T, S))
    # Dispatch to another function here to avoid potential infinite recursion
    _loess(convert(AbstractMatrix{R}, xs), convert(AbstractVector{R}, ys); kwargs...)
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
function predict(model::LoessModel{T}, z::Number) where T
    adjacent_verts = traverse(model.kdtree, (T(z),))

    @assert(length(adjacent_verts) == 2)
    v₁, v₂ = adjacent_verts[1][1], adjacent_verts[2][1]

    if z == v₁ || z == v₂
        return first(model.predictions_and_gradients[[z]])
    end

    y₁, dy₁ = model.predictions_and_gradients[[v₁]]
    y₂, dy₂ = model.predictions_and_gradients[[v₂]]

    b_int = cubic_interpolation(v₁, y₁, dy₁, v₂, y₂, dy₂)

    return evalpoly(z, b_int)
end

function predict(model::LoessModel, zs::AbstractVector)
    if size(model.xs, 2) > 1
        throw(ArgumentError("multivariate blending not yet implemented"))
    end

    return [predict(model, z) for z in zs]
end

function predict(model::LoessModel, zs::AbstractMatrix)
    if size(model.xs, 2) != size(zs, 2)
        throw(DimensionMismatch("number of columns in input matrix must match the number of columns in the model matrix"))
    end

    if size(zs, 2) == 1
        return predict(model, vec(zs))
    else
        return [predict(model, row) for row in eachrow(zs)]
    end
end

fitted(model::LoessModel) = predict(model, modelmatrix(model))

residuals(model::LoessModel) = response(model) .- fitted(model)

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
    cubic_interpolation(x₁, y₁, dy₁, x₂, y₂, dy₂)

Compute the coefficients of the cubic polynomial ``f`` for which
```math
\begin{aligned}
    y₁  &= f(x₁)  \\
    dy₁ &= f'(x₁) \\
    y₂  &= f(x₂)  \\
    dy₂ &= f'(x₂) \\
\end{aligned}
```
"""
function cubic_interpolation(x₁, y₁, dy₁, x₂, y₂, dy₂)
    Δx = x₁ - x₂
    Δx³ = Δx^3
    Δy = y₁ - y₂
    num0 = -x₂ * (x₁ * Δx * (dy₂ * x₁ + dy₁ * x₂) + x₂ * (x₂ - 3 * x₁) * y₁) + x₁^2 * (x₁ - 3 * x₂) * y₂
    num1 = dy₂ * x₁ * Δx * (x₁ + 2 * x₂) - x₂ * (dy₁ * (x₁ * x₂ + x₂^2 - 2 * x₁^2) + 6 * x₁ * Δy)
    num2 = -(dy₁ * Δx * (x₁ + 2 * x₂)) + dy₂ * (x₁ * x₂ + x₂^2 - 2 * x₁^2) + 3 * (x₁ + x₂) * Δy
    num3 = (dy₁ + dy₂) * Δx - 2 * Δy
    return (
        num0 / Δx³,
        num1 / Δx³,
        num2 / Δx³,
        num3 / Δx³
    )
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
