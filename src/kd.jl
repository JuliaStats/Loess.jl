using Compat

# Simple static kd-trees.


abstract KDNode


immutable KDTree{T <: FloatingPoint}
	# A matrix of n, m-dimensional observations
	xs::AbstractMatrix{T}

	# Permutaiton of the data used by the structure (to avoid sorting or
	# otherwise modifying xs)
	perm::Vector{Int}

	root::KDNode

	verts::Set{Vector{T}}

	# Top-level bounding box
	bounds::Matrix{T}
end


# Construct a kd-tree
#
# Args:
#   xs: Data organized in the tree.
#   leaf_size_factor: Stop spliting if a node contains
#       fewer than leaf_size_factor * n elements.
#   leaf_diameter_factor: Stop spliting if a node's bounding
#       hypercube has diameter less that leaf_diameter_factor
#       times the diameter of the root node's bounding hypercube.
#
# Returns:
#   A KDTree object
#
function KDTree{T <: FloatingPoint}(xs::AbstractMatrix{T},
	                                leaf_size_factor=0.05,
	                                leaf_diameter_factor=0.0)
	n, m = size(xs)
	perm = collect(1:n)

	bounds = Array(T, (2, m))
	for j in 1:m
		col = xs[:,j]
		bounds[1, j] = minimum(col)
		bounds[2, j] = maximum(col)
	end
	diam = diameter(bounds)

	leaf_size_cutoff = ceil(Integer, (leaf_size_factor * n))
	leaf_diameter_cutoff = leaf_diameter_factor * diam
	verts = Set{Vector{T}}()

	# Add verticies defined by the bounds
	for vert in product([bounds[:,j] for j in 1:m]...)
		push!(verts, T[vert...])
	end

	root = build_kdtree(xs, sub(perm, 1:n), bounds,
		                leaf_size_cutoff, leaf_diameter_cutoff, verts)

	KDTree(xs, collect(1:n), root, verts, bounds)
end


immutable KDLeafNode <: KDNode
end


immutable KDInternalNode{T <: FloatingPoint} <: KDNode
	j::Int # dimension on which the data is split
	med::T # median value where the split occours
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
	euclidean(vec(bounds[1,:]), vec(bounds[2,:]))
end


# Recursively build a kd-tree
#
# Args:
#   xs: Data being orginized.
#   perm: Permutation of the data, used to avoid
#       directly sorting or modifying xs.
#   bounds: Bounding hypercube of the node.
#   leaf_size_cutoff: stop spliting on nodes with more than
#       this many values.
#   leaf_diameter_cutoff: stop splitting on nodes with less
#        than this diameter.
#   verts: current set of vertexes
#
# Modifies:
#   perm, verts
#
# Returns:
#   Either a KDLeafNode or a KDInternalNode
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
	if isodd(length(perm))
		mid = div(length(perm), 2)
		select!(perm, mid, by=i -> xs[i, j])
		med = xs[perm[mid], j]
		mid1 = mid
		mid2 = mid + 1
	else
		mid1 = div(length(perm), 2)
		mid2 = mid1 + 1
		select!(perm, mid1, by=i -> xs[i, j])
		select!(perm, mid2, by=i -> xs[i, j])
		med = (xs[perm[mid1], j] + xs[perm[mid2], j]) / 2
	end

	leftbounds = copy(bounds)
	leftbounds[2, j] = med
	leftnode = build_kdtree(xs, sub(perm, 1:mid1), bounds,
		                    leaf_size_cutoff, leaf_diameter_cutoff, verts)

	rightbounds = copy(bounds)
	rightbounds[1, j] = med
	rightnode = build_kdtree(xs, sub(perm, mid2:length(perm)), bounds,
		                     leaf_size_cutoff, leaf_diameter_cutoff, verts)

	coords = Array(Array, m)
	for i in 1:m
		if i == j
			coords[i] = [med]
		else
			coords[i] = bounds[:, i]
		end
	end

	for vert in product(coords...)
		push!(verts, T[vert...])
	end

	KDInternalNode{T}(j, med, leftnode, rightnode)
end


# Given a bounding hypecube, return its verticies
function bounds_verts(bounds::Matrix)
	collect(product([bounds[:, i] for i in 1:size(bounds, 2)]...))
end


# Traverse the tree to the bottom and return the verticies of
# the leaf node's bounding hypercube.
function traverse{T}(kdtree::KDTree{T}, xs::AbstractVector{T})
	m = size(kdtree.bounds, 2)

	if length(xs) != m
		error("$(m)-dimensional kd-tree searched with a length $(length(x)) vector.")
	end

	for j in 1:m
		if xs[j] < kdtree.bounds[1, j] || xs[j] > kdtree.bounds[2, j]
			error(
				"""
				Loess cannot perform extrapolation. Predict can only be applied
				to points within the bounding hypercube of the data used to train
				the model.
				""")
		end
	end

	bounds = copy(kdtree.bounds)
	node = kdtree.root
	while !isa(node, KDLeafNode)
		if xs[node.j] <= node.med
			bounds[2, node.j] = node.med
			node = node.leftnode
		else
			bounds[1, node.j] = node.med
			node = node.rightnode
		end
	end

	bounds_verts(bounds)
end

