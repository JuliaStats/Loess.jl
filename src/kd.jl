# Simple static kd-trees.

abstract type KDNode end

struct KDLeafNode <: KDNode
end

struct KDInternalNode{T <: AbstractFloat} <: KDNode
    j::Int             # dimension on which the data is split
    med::T             # median value where the split occours
    leftnode::KDNode
    rightnode::KDNode
end


struct KDTree{T <: AbstractFloat}
    xs::AbstractMatrix{T} # A matrix of n, m-dimensional observations
    perm::Vector{Int}     # permutation of data to avoid modifying xs
    root::KDNode          # root node
    verts::Set{Vector{T}}
    bounds::Matrix{T}     # Top-level bounding box
end


"""
    KDTree(xs, leaf_size_factor, leaf_diameter factor)

Construct a kd-tree

Args:
  - `xs`: an `n` x `m` matrix containing `n`, `m`-dimensional observations.
  - `leaf_size_factor`: Stop spliting if a node contains
      fewer than `leaf_size_factor * n` elements.
  - `leaf_diameter_factor`: Stop spliting if a node's bounding
      hypercube has diameter less that `leaf_diameter_factor`
      times the diameter of the root node's bounding hypercube.

Returns:
  A `KDTree` object

"""
function KDTree(xs::AbstractMatrix{T},
                leaf_size_factor=0.05,
                leaf_diameter_factor=0.0) where T <: AbstractFloat

    n, m = size(xs)
    perm = collect(1:n)

    bounds = Array{T}(undef, 2, m)
    for j in 1:m
        col = xs[:,j]
        bounds[1, j] = minimum(col)
        bounds[2, j] = maximum(col)
    end

    diam = diameter(bounds)

    leaf_size_cutoff = ceil(Int, leaf_size_factor * n)
    leaf_diameter_cutoff = leaf_diameter_factor * diam
    verts = Set{Vector{T}}()

    # Add a vertex for each corner of the hypercube
    for vert in Iterators.product([bounds[:,j] for j in 1:m]...)
        push!(verts, T[vert...])
    end

    root = build_kdtree(xs, perm, bounds, leaf_size_cutoff, leaf_diameter_cutoff, verts)

    KDTree(xs, collect(1:n), root, verts, bounds)
end




"""
    diameter(bounds)

Compute the diamater of a hypercube (i.e. the maximum distance between any 2 points in the hypercube) defined by `bounds`.

Args:
  - `bounds`: A 2 by `n` matrix where `bounds[1,i]` gives the lower bound in
      the `i`th dimension and `bounds[2,j]` the upper bound.

Returns:
  Computed diameter

"""
function diameter(bounds::Matrix)
    euclidean(vec(bounds[1,:]), vec(bounds[2,:]))
end


"""
    build_kdtree(xs, perm, bounds, leaf_size_cutoff, leaf_diameter_cutoff, verts)

Recursively build a kd-tree

Args:
  - `xs`: Data being orginized.
  - `perm`: Permutation of the data, used to avoid
      directly sorting or modifying `xs`.
  - `bounds`: Bounding hypercube of the node.
  - `leaf_size_cutoff`: stop spliting on nodes with more than
      this many values.
  - `leaf_diameter_cutoff`: stop splitting on nodes with less
       than this diameter.
  - `verts`: current set of vertexes

Modifies:
  `perm`, `verts`

Returns:
  Either a `KDLeafNode` or a `KDInternalNode`
"""
function build_kdtree(xs::AbstractMatrix{T},
                      perm::AbstractArray,
                      bounds::Matrix{T},
                      leaf_size_cutoff::Int,
                      leaf_diameter_cutoff::T,
                      verts::Set{Vector{T}}) where T
    n, m = size(xs)

    if length(perm) <= leaf_size_cutoff || diameter(bounds) <= leaf_diameter_cutoff
        return KDLeafNode()
    end

    # split on the dimension with the largest spread
    # maxspread, j = findmax(maximum(xs[perm, k]) - minimum(xs[perm, k]) for k in 1:m)
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
        mid = length(perm) ÷ 2
        partialsort!(perm, mid, by=i -> xs[i, j])
        med = xs[perm[mid], j]
        mid1 = mid
        mid2 = mid + 1
    else
        mid1 = length(perm) ÷ 2
        mid2 = mid1 + 1
        partialsort!(perm, mid1:mid2, by=i -> xs[i, j])
        med = (xs[perm[mid1], j] + xs[perm[mid2], j]) / 2
    end

    leftbounds = copy(bounds)
    leftbounds[2, j] = med
    leftnode = build_kdtree(xs, view(perm,1:mid1), leftbounds,
                            leaf_size_cutoff, leaf_diameter_cutoff, verts)

    rightbounds = copy(bounds)
    rightbounds[1, j] = med
    rightnode = build_kdtree(xs, view(perm,mid2:length(perm)), rightbounds,
                             leaf_size_cutoff, leaf_diameter_cutoff, verts)

    coords = Array{Array}(undef, m)
    for i in 1:m
        if i == j
            coords[i] = [med]
        else
            coords[i] = bounds[:, i]
        end
    end

    for vert in Iterators.product(coords...)
        push!(verts, T[vert...])
    end

    KDInternalNode{T}(j, med, leftnode, rightnode)
end


"""
    bounds_verts(bounds)

Given a bounding hypecube `bounds`, return its verticies
"""
function bounds_verts(bounds::Matrix)
    collect(Iterators.product([bounds[:, i] for i in 1:size(bounds, 2)]...))
end


"""
    traverse(kdtree, x)

Traverse the tree `kdtree` to the bottom and return the verticies of
the bounding hypercube of the leaf node containing the point `x`.
"""
function traverse(kdtree::KDTree{T}, x::AbstractVector{T}) where T
    m = size(kdtree.bounds, 2)

    if length(x) != m
        throw(DimensionMismatch("$(m)-dimensional kd-tree searched with a length $(length(x)) vector."))
    end

    for j in 1:m
        if x[j] < kdtree.bounds[1, j] || x[j] > kdtree.bounds[2, j]
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
        if x[node.j] <= node.med
            bounds[2, node.j] = node.med
            node = node.leftnode
        else
            bounds[1, node.j] = node.med
            node = node.rightnode
        end
    end

    bounds_verts(bounds)
end
