

require("Distance")
require("Iterators")


module Loess

import Iterators.product
import Distance.euclidean

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


function loess{T <: FloatingPoint}(xs::AbstractVector{T}, ys::AbstractVector{T};
	                            	   normalize::Bool=true, span::T=0.75)
	loess(reshape(xs, (length(xs), 1)), ys, normalize=normalize, span=span)
end


function loess{T <: FloatingPoint}(xs::AbstractMatrix{T}, ys::AbstractVector{T};
	                               normalize::Bool=true, span::T=0.75)
	if size(xs, 1) != size(ys, 1)
		error("Predictor and response arrays must of the same length")
	end

	n, m = size(xs)
	q = iceil(span * n)

	if normalize && m > 1
		xs = copy(xs)
		normalize!(xs)
	end

	kdtree = KDTree(xs)
	verts = Array(T, (length(kdtree.verts), m))

	# map verticies to their index in the bs coefficient matrix
	verts = Dict{Vector{T}, Int}()
	for (k, vert) in enumerate(kdtree.verts)
		verts[vert] = k
	end

	# Fit each vertex
	ds = Array(T, n) # distances
	perm = collect(1:n)

	bs = Array(T, (length(kdtree.verts), 1 + m))

	# TODO: higher degree fitting
	us = Array(T, (q, 1 + m))
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
			us[i,1] = 1
			d = sqrt(tricubic(ds[perm[i]] / dmax))
			us[i,2:end] = xs[perm[i],:] * d
			vs[i] = ys[perm[i]] * d
		end
		F = qrfact!(us)
		Q = full(F[:Q], true)[:,1:m+1]
		R = F[:R][1:m+1, 1:m+1]
		bs[k,:] = R \ (Q' * vs)
	end

	LoessModel{T}(xs, ys, bs, verts, kdtree)
end


# Evaluate a multivariate polynomial with coefficients bs
function evalpoly(xs, bs)
	m = length(xs)
	degree = (length(bs) - 1) / m
	y = 0.0
	for i in 1:m
		yi = 0.0
		for j in (1 + i * m):-1:(2 + (i-1) * m)
			yi = yi * xs[i] + bs[j]
		end
		y += yi * xs[i]
	end
	y + bs[1]
end


function predict{T <: FloatingPoint}(model::LoessModel{T}, z::T)
	predict(model, T[z])
end


function predict{T <: FloatingPoint}(model::LoessModel{T}, zs::AbstractVector{T})
	m = size(model.bs, 2) - 1

	if length(zs) != m
		error("$(size(model.bs, 2))-dimensional model applied to length $(length(zs)) vector")
	end

	adjacent_verts = traverse(model.kdtree, zs)

	if m == 1
		@assert(length(adjacent_verts) == 2)
		z = zs[1]
		u = (z - adjacent_verts[1][1]) /
			    (adjacent_verts[2][1] - adjacent_verts[1][1])

		y1 = evalpoly(zs, model.bs[model.verts[[adjacent_verts[1][1]]],:])
		y2 = evalpoly(zs, model.bs[model.verts[[adjacent_verts[2][1]]],:])
		return u * y1 + (1.0 - u) * y2
	else
		error("Multivariate blending not yet implemented")
		# TODO:
		#   1. Univariate linear interpolation between adjacent verticies.
		#   2. Blend these estimates. (I'm not sure how this is done.)
	end
end


# Tricubic weight function
#
# Args:
#   u: Position between 0 and 1
#
function tricubic(u)
	(1 - u^3)^3
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
	tmp = Array()
	for j in 1:m
		copy!(tmp, xs[:,j])
		sort!(tmp)
		xs[:,j] ./= mean(tmp[cut+1:n-cut])
	end
end




end
