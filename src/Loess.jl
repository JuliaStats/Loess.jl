module Loess

import Distances: euclidean
import StatsAPI: fitted, modelmatrix, predict, residuals, response

using Statistics, LinearAlgebra
using Distributions: Normal, TDist, quantile

export loess, fitted, modelmatrix, predict, residuals, response

include("kd.jl")


struct LoessModel{T <: AbstractFloat}
    # An n by m predictor matrix containing n observations from m predictors
    xs::Matrix{T}
    # A length n response vector
    ys::Vector{T}
    # kd-tree vertexes mapped to prediction and gradient at each vertex
    predictions_and_gradients::Dict{Vector{T}, Vector{T}}
    # kd-tree vertexes mapped to generating vectors for prediction (first row)
    # and gradient (second row) at each vertex. The values values of the
    # predictions_and_gradients field are the values of hatmatrix_generator
    # multiplied by ys
    hatmatrix_generator::Dict{Vector{T}, Matrix{T}}
    # The kd-tree
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
    vs = Array{T}(undef, q)
    # The hat matrix and the derivative at the vertices of kdtree
    F = Dict{Vector{T},Matrix{T}}()


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

        # We center the predictors since this will greatly simplify subsequent
        # calculations. With centerting, the intercept is the prediction at the
        # the vertex and the derivative is linear coefficient.
        #
        # We still only support m == 1 so we hard code for m=1 below. If support
        # for more predictors is introduced then this code would have to be adjusted.
        #
        # See Cleveland and Grosse page 54 column 1. The results assume centering
        # but I'm not sure, the centering is explicitly stated in the paper.
        x₁ = xs[perm[1], 1]
        for i in 1:q
            pᵢ = perm[i]
            w = sqrt(tricubic(ds[pᵢ] / dmax))
            us[i, 1] = w
            for j in 1:m # we still only support m == 1
                x = xs[pᵢ, j] - x₁ # center
                wxl = w
                for l in 1:degree
                    wxl *= x
                    us[i, 1 + (j - 1)*degree + l] = wxl # w*x^l
                end
            end
            vs[i] = ys[pᵢ] * w
        end

        Fact = svd(us)

        Ftmp = (Fact \ Diagonal(us[:, 1]))[1:2, :]
        FF = zeros(T, 2, n)
        FF[:, perm[1:q]] = Ftmp
        F[vert] = FF

        coefs = Fact \ vs

        predictions_and_gradients[vert] = coefs[1:2]
    end

    LoessModel(
        convert(Matrix{T}, xs),
        convert(Vector{T}, ys),
        predictions_and_gradients,
        F,
        kdtree,
    )
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
function predict(model::LoessModel{T}, x::Number) where T
    adjacent_verts = traverse(model.kdtree, (T(x),))

    @assert(length(adjacent_verts) == 2)
    v₁, v₂ = adjacent_verts[1][1], adjacent_verts[2][1]

    if x == v₁ || x == v₂
        return first(model.predictions_and_gradients[[x]])
    end

    y₁, dy₁ = model.predictions_and_gradients[[v₁]]
    y₂, dy₂ = model.predictions_and_gradients[[v₂]]

    b_int = cubic_interpolation(v₁, y₁, dy₁, v₂, y₂, dy₂)

    return evalpoly(x, b_int)
end

struct LoessPrediction{T}
    predictions::Vector{T}
    lower::Vector{T}
    upper::Vector{T}
    sₓ::Vector{T}
    δ₁::T
    δ₂::T
    s::T
    ρ::T
end

"""
    predict(
        model::LoessModel,
        x::AbstractVecOrMat = modelmatrix(model);
        interval::Union{Symbol, Nothing}=nothing,
        level::Real=0.95
    )

Compute predictions from the fitted Loess `model` at the values in `x`.
By default, only the predictions are returned. If `interval` is set to
:confidence` then a `LoessPrediction` struct is returned which contains
upper and lower confidence bounds at `level` as well the the quantities
used for computing the confidence bounds.

When the prodictor takes values at the edges of the KD tree, the predictions
are already computed and the function is simply a look-up. For other values
of the preditor, a cubic spline approximation is used for the prediction.

When confidence bounds are requested, a matrix of size `n \times n`` is
constructed where ``n`` is the length of the fitted data vector. Hence,
this caluculation is only feasible when ``n`` is not too large. For details
on the calculations, see the cited reference.

Reference: Cleveland, William S., and Eric Grosse. "Computational methods
for local regression." Statistics and computing 1, no. 1 (1991): 47-62.
"""
function predict(
    model::LoessModel,
    x::AbstractVecOrMat = modelmatrix(model);
    interval::Union{Symbol, Nothing}=nothing,
    level::Real=0.95
)
    if size(model.xs, 2) > 1
        throw(ArgumentError("multivariate blending not yet implemented"))
    end

    if interval !== nothing && interval !== :confidence
        if interval === :prediction
            throw(ArgumentError("predictions not implemented. If you know how to compute them then please file an issue and explain how."))
        end
        throw(ArgumentError("interval must be either :prediction or :confidence but was $interval"))
    end

    if !(0 < level < 1)
        throw(ArgumentError("level must be between zero and one but was $level"))
    end

    predictions = [predict(model, _x) for _x in x]

    if interval === nothing
        return predictions
    else
        # see Cleveland and Grosse 1991 p.50.
        L = hatmatrix(model)
        L̄ = L - I
        L̄L̄ = L̄' * L̄
        δ₁ = tr(L̄L̄)
        δ₂ = sum(abs2, L̄L̄)
        ε̂ = L̄ * model.ys
        s = sqrt(sum(abs2, ε̂) / δ₁)
        ρ = δ₁^2 / δ₂
        qt = quantile(TDist(ρ), (1 + level) / 2)
        sₓ = [s*sqrt(sum(abs2, _hatmatrix_x(model, _x))) for _x in x]
        lower = [_x - qt * _sₓ for (_x, _sₓ) in zip(predictions, sₓ)]
        upper = [_x + qt * _sₓ for (_x, _sₓ) in zip(predictions, sₓ)]
        return LoessPrediction(predictions, lower, upper, sₓ, δ₁, δ₂, s, ρ)
    end
end

fitted(model::LoessModel) = predict(model, modelmatrix(model))

residuals(model::LoessModel) = response(model) .- fitted(model)

function hatmatrix(model::LoessModel{T}) where T
    n = length(model.ys)
    L = zeros(T, n, n)
    for (i, r) in enumerate(eachrow(model.xs))
        z = only(r) # we still only support one predictor
        L[i, :] = _hatmatrix_x(model, z)
    end
    return L
end

# Compute elements of the hat matrix of `model` at `x`.
# See Cleveland and Grosse 1991 p. 54.
function _hatmatrix_x(model::LoessModel{T}, x::Number) where T
    n = length(model.ys)

    adjacent_verts = traverse(model.kdtree, (T(x),))

    @assert(length(adjacent_verts) == 2)
    v₁, v₂ = adjacent_verts[1], adjacent_verts[2]

    if x == v₁ || x == v₂
        Lx = model.hatmatrix_generator[[x]][1, :]
    else
        Lx = zeros(T, n)
        for j in 1:n
            b_int = cubic_interpolation(
                v₁,
                model.hatmatrix_generator[[v₁]][1, j],
                model.hatmatrix_generator[[v₁]][2, j],
                v₂,
                model.hatmatrix_generator[[v₂]][1, j],
                model.hatmatrix_generator[[v₂]][2, j],
            )
            Lx[j] = evalpoly(x, b_int)
        end
    end
    return Lx
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
