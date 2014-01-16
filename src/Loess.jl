

require("Distance")
require("Iterators")


module Loess

import Iterators.product
import Distance.euclidean

export loess, predict

include("kd.jl")


type LoessModel{T <: FloatingPoint}
	# An n by m predictor matrix containing n observations from m predictors
	xs::AbstractMatrix{T}

	# A length n response vector
	ys::AbstractVector{T}

	# Least squares coefficients
	bs::Matrix{T}

	# kd-tree vertexes mapped to indexes
	verts::Dict{Vector{T}, Int}

	kdtree::KDTree{T}
end


# Fit a loess model.
#
# Args:
#   xs: A n by m matrix with n observations from m independent predictors
#   ys: A length n response vector.
#   normalize: Normalize the scale of each predicitor. (default true when m > 1)
#   span: The degree of smoothing, typically in [0,1]. Smaller values result in smaller
#       local context in fitting.
#   degree: Polynomial degree.
#
# Returns:
#   A fit LoessModel.
#
function loess{T <: FloatingPoint}(xs::AbstractVector{T}, ys::AbstractVector{T};
	                            	   normalize::Bool=true, span::T=0.75, degree::Int=2)
	loess(reshape(xs, (length(xs), 1)), ys, normalize=normalize, span=span, degree=degree)
end


function loess{T <: FloatingPoint}(xs::AbstractMatrix{T}, ys::AbstractVector{T};
	                               normalize::Bool=true, span::T=0.75, degree::Int=2)
	if size(xs, 1) != size(ys, 1)
		error("Predictor and response arrays must of the same length")
	end

	n, m = size(xs)
	q = iceil(span * n)

	# TODO: We need to keep track of how we are normalizing so we can
	# corerctly apply predict to unnormalized data. We should have a normalize
	# function that just returns a vector of scaling factors.
	if normalize && m > 1
		xs = copy(xs)
		normalize!(xs)
	end

	kdtree = KDTree(xs, 0.05 * span)
	verts = Array(T, (length(kdtree.verts), m))

	# map verticies to their index in the bs coefficient matrix
	verts = Dict{Vector{T}, Int}()
	for (k, vert) in enumerate(kdtree.verts)
		verts[vert] = k
	end

	# Fit each vertex
	ds = Array(T, n) # distances
	perm = collect(1:n)

	bs = Array(T, (length(kdtree.verts), 1 + degree * m))

	# TODO: higher degree fitting
	us = Array(T, (q, 1 + degree * m))
	vs = Array(T, q)
	for (vert, k) in verts
		for i in 1:n
			perm[i] = i
		end

		for i in 1:n
			ds[i] = euclidean(vec(vert), vec(xs[i,:]))
		end

		# copy the q nearest points to vert into X
		select!(perm, q, by=i -> ds[i])
		dmax = 0.0
		for i in 1:q
			dmax = max(dmax, ds[perm[i]])
		end

		for i in 1:q
			w = tricubic(ds[perm[i]] / dmax)
			us[i,1] = w
			for j in 1:m
				x = xs[perm[i], j]
				xx = x
				for l in 1:degree
					us[i, 1 + (j-1)*degree + l] = w * xx
					xx *= x
				end
			end
			vs[i] = ys[perm[i]] * w
		end
		F = qrfact!(us)
		Q = full(F[:Q])[:,1:degree*m+1]
		R = F[:R][1:degree*m+1, 1:degree*m+1]
		bs[k,:] = R \ (Q' * vs)
	end

	LoessModel{T}(xs, ys, bs, verts, kdtree)
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
function predict{T <: FloatingPoint}(model::LoessModel{T}, z::T)
	predict(model, T[z])
end


function predict{T <: FloatingPoint}(model::LoessModel{T}, zs::AbstractVector{T})
	m = size(model.xs, 2)

	# in the univariate case, interpret a non-singleton zs as vector of
	# ponits, not one point
	if m == 1 && length(zs) > 1
		return predict(model, reshape(zs, (length(zs), 1)))
	end

	if length(zs) != m
		error("$(m)-dimensional model applied to length $(length(zs)) vector")
	end

	adjacent_verts = traverse(model.kdtree, zs)

	if m == 1
		@assert(length(adjacent_verts) == 2)
		z = zs[1]
		u = (z - adjacent_verts[1][1]) /
			    (adjacent_verts[2][1] - adjacent_verts[1][1])

		y1 = evalpoly(zs, model.bs[model.verts[[adjacent_verts[1][1]]],:])
		y2 = evalpoly(zs, model.bs[model.verts[[adjacent_verts[2][1]]],:])
		return (1.0 - u) * y1 + u * y2
	else
		error("Multivariate blending not yet implemented")
		# TODO:
		#   1. Univariate linear interpolation between adjacent verticies.
		#   2. Blend these estimates. (I'm not sure how this is done.)
	end
end


function predict{T <: FloatingPoint}(model::LoessModel{T}, zs::AbstractMatrix{T})
	ys = Array(T, size(zs, 1))
	for i in 1:size(zs, 1)
		ys[i] = predict(model, vec(zs[i,:]))
	end
	ys
end


# Tricubic weight function
#
# Args:
#   u: Distance between 0 and 1
#
# Returns:
#   A weighting of the distance u
#
function tricubic(u)
	(1 - u^3)^3
end


# Evaluate a multivariate polynomial with coefficients bs
function evalpoly(xs, bs)
	m = length(xs)
	degree = div(length(bs) - 1, m)
	y = 0.0
	for i in 1:m
		yi = 0.0
		x = xs[i]
		xx = x
		for l in 1:degree
			y += xx * bs[i, 1 + (i-1)*degree + l]
			xx *= x
		end
	end
	y + bs[1]
end


# Default normalization procedure for predictors.
#
# This simply normalizes by the mean of everything between the 10th an 90th percentiles.
#
# Args:
#   xs: a matrix of predictors
#   q: cut the ends of at quantiles q and 1-q
#
# Modifies:
#   xs
#
function normalize!{T <: FloatingPoint}(xs::AbstractMatrix{T}, q::T=0.100000000000000000001)
	n, m = size(xs)
	cut = iceil(q * n)
	tmp = Array(T, n)
	for j in 1:m
		copy!(tmp, xs[:,j])
		sort!(tmp)
		xs[:,j] ./= mean(tmp[cut+1:n-cut])
	end
end


end
