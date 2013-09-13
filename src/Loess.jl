

module Loess


type LoessModel{T <: FloatingPoint}
	# An n by m predictor matrix containing n observations from m predictors
	xs::AbstractMatrix{T}

	# A length n response vector
	ys::AbstractVector{T}


end


function loess{T <: FloatingPoint}(xs::AbstractMatrix{T}, ys::AbstractVector{T};
	                               normalize::Bool=true)
	if size(xs, 1) != size(ys, 1)
		error("Predictor and response arrays must of the same length")
	end

	LoessModel(xs, ys)
end


function predict(model::LoessModel, zs::AbstractMatrix{T})
	# TODO
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
