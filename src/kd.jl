

# Simple static kd-trees.


abstract KDNode


immutable KDTree{T <: FloatingPoint}
	# A matrix of n, m-dimensional observations
	xs::AbstractMatrix{T}

	# Permutaiton of the data used by the structure (to avoid sorting or
	# otherwise modifying xs)
	perm::Vector{Int}

	root::KDNode
end


function KDTree{T <: FloatingPoint}(xs::AbstractMatrix{T})
	n, m = size(xs)
	perm = collect(1:n)

	bounds = Array(T, (2, m))
	for j in 1:m
		col = xs[:,j]
		bounds[1, j] = min(col)
		bounds[2, j] = max(col)
	end
	diam = diameter(bounds)

	# TODO: pass in values for these
	leaf_size_cutoff = iceil(0.05 * n)
	leaf_diameter_cutoff = 0.0 * diam
	verts = Set{Vector{T}}()

	root = build_kdtree(xs, sub(perm, 1:n), bounds,
		                leaf_size_cutoff, leaf_diameter_cutoff, verts)

	KDTree(xs, collect(1:n), root)
end


immutable KDLeafNode <: KDNode
end


immutable KDInternalNode <: KDNode
	j::Int # dimension on which the data is split
	leftnode::KDNode
	rightnode::KDNode
end



# Compute the diamater of a hypercube defined by bounds
#
# Args:
#   bounds: A 2 by n matrix where bounds[1,i] gives the lower bound in
#           the ith dimension and bonuds[2,j] the upper.
#
# Returns:
#   Computed diameter
#
function diameter(bounds::Matrix)
	diam = 0.0
	for i in 1:size(bounds, 2)
		diam += (bounds[2, i] - bounds[1, i])^2
	end
	sqrt(diam)
end


# Recursively build a kd-tree
#
function build_kdtree{T}(xs::AbstractMatrix{T},
	                     perm::SubArray,
	                     bounds::Matrix{T},
	                     leaf_size_cutoff::Int,
	                     leaf_diameter_cutoff::T,
	                     verts::Set{Vector{T}})
	n, m = size(xs)

	if length(perm) <= leaf_size_cutoff || diameter(bounds) <= leaf_diameter_cutoff
		return KDLeafNode()
	end

	# split on the dimension with the largest spread
	j = 1
	maxspread = 0
	for k in 1:m
		xmin = Inf
		xmax = -Inf
		for i in perm
			xmin = min(xmin, xs[i, k])
			xmax = max(xmax, xs[i, k])
		end
		if xmax - xmin > maxspread
			maxspread = xmax - xmin
			j = k
		end
	end

	# find the median and partition
	mid = div(length(perm), 2)
	select!(perm, mid, by=i -> xs[i, j])
	med = xs[perm[mid], j]

	leftbounds = copy(bounds)
	leftbounds[2, j] = med
	leftnode = build_kdtree(xs, sub(perm, 1:mid), bounds,
		                    leaf_size_cutoff, leaf_diameter_cutoff, verts)

	rightbounds = copy(bounds)
	rightbounds[1, j] = med
	rightnode = build_kdtree(xs, sub(perm, mid+1:length(perm)), bounds,
		                     leaf_size_cutoff, leaf_diameter_cutoff, verts)

	KDInternalNode(j, leftnode, rightnode)

	# TODO: We need to keep track of verticies also. I'm not sure what else we
	# need to find all points within a particular range.
end

