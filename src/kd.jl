# Simple static kd-trees.

struct KDNode{T <: AbstractFloat}
    j::Int             # dimension on which the data is split
    med::T             # median value where the split occours
    leftnode::Union{Nothing, KDNode{T}}
    rightnode::Union{Nothing, KDNode{T}}
end


struct KDTree{T <: AbstractFloat}
    xs::Matrix{T}         # A matrix of n, m-dimensional observations
    perm::Vector{Int}     # permutation of data to avoid modifying xs
    root::KDNode{T}       # root node
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
function KDTree(
    xs::AbstractMatrix{T},
    leaf_size_factor::Real,
    leaf_diameter_factor::Real
) where T <: AbstractFloat

    n, m = size(xs)
    perm = collect(1:n)

    bounds = Array{T}(undef, 2, m)
    for j in 1:m
        col = xs[:,j]
        bounds[1, j] = minimum(col)
        bounds[2, j] = maximum(col)
    end

    diam = diameter(bounds)

    leaf_size_cutoff = floor(Int, leaf_size_factor * n)
    leaf_diameter_cutoff = leaf_diameter_factor * diam
    verts = Set{Vector{T}}()

    # Add a vertex for each corner of the hypercube
    # This deviates from the original implementation from the paper where the outer verted were
    # made a bit wider than the data limits.
    for vert in Iterators.product([bounds[:,j] for j in 1:m]...)
        push!(verts, T[vert...])
    end

    root = build_kdtree(xs, perm, bounds, leaf_size_cutoff, leaf_diameter_cutoff, verts)

    KDTree(convert(Matrix{T}, xs), collect(1:n), root, verts, bounds)
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
  Either a `nothing` or a `KDNode`
"""
function build_kdtree(xs::AbstractMatrix{T},
                      perm::AbstractVector,
                      bounds::Matrix{T},
                      leaf_size_cutoff::Real,
                      leaf_diameter_cutoff::Real,
                      verts::Set{Vector{T}}) where T

    Base.require_one_based_indexing(xs)
    Base.require_one_based_indexing(perm)

    n, m = size(xs)

    if length(perm) <= leaf_size_cutoff || diameter(bounds) <= leaf_diameter_cutoff
        @debug "Creating leaf node" length(perm) leaf_size_cutoff diameter(bounds) leaf_diameter_cutoff
        return nothing
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

    # Find the "median" and partition
    #
    # The aim of the algorithm is to split the data recursively in two roughly equally sized
    # subsets. To do so, we'll use the median element of x[:, j] but there are a
    # few corner cases that require some care: sets with an even number of elements
    # doesn't have a unique median and ties. Below, we'll list the possibilities.
    #
    # - Odd number of elements and no ties, e.g. [1, 2, 3]: the median is
    #   unambiguously 2 and split the data into [1, 2] and [3]
    #
    # - Even number of element and no ties, e.g. [1, 2, 3, 4]: we choose the left middle
    #   value 2 as median and split the data into [1, 2] and [3, 4]
    #
    # - Ties will cause a search to the change in value that divides the set most evenly.
    #   E.g. [1, 2, 2, 3] uses 2 as median value to split the data into [1, 2, 2] and [3]
    #   but [1, 1, 2, 2, 2] uses 1 to split into [1, 1] and [2, 2, 2] even though 1 is
    #   not a proper median value. This avoids that the same value is in two buckets.
    #
    # The details here are reversed engineered from the C/Fortran implementation wrapped
    # by R and also distribtued on NETLIB.
    mid = (length(perm) + 1) ÷ 2
    @debug "Candidate median index and median value" mid xs[perm[mid], j]

    offset = 0
    local mid1, mid2
    while true
        mid1 = mid + offset
        mid2 = mid1 + 1
        if mid1 < 1
            @debug "mid1 is zero. All elements are identical. Creating vertex and then two leaves" mid1 length(perm) xs[perm[mid], j]
            offset = mid1 = 0
            mid2 = length(perm) + 1
            break
        end
        if mid2 > length(perm)
            @debug "mid2 is out of bounds. Continuing with negative offset" mid2 length(perm) offset
            # This makes the offset 0, 1, -1, 2, -2, ...
            offset = -offset + (offset <= 0)
            continue
        end
        p12 = partialsort!(perm, mid1:mid2, by = i -> xs[i, j])
        if xs[p12[1], j] == xs[p12[2], j]
            @debug "tie! Adjusting offset" xs[p12[1], j] xs[p12[2], j] offset
            # This makes the offset 0, 1, -1, 2, -2, ...
            offset = -offset + (offset <= 0)
        else
            break
        end
    end
    mid += offset
    med = xs[perm[mid], j]
    @debug "Accepted median index and median value" mid med

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

    KDNode(j, med, leftnode, rightnode)
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
function traverse(kdtree::KDTree{T}, x::NTuple{N,T}) where {N,T}

    m = size(kdtree.bounds, 2)

    if N != m
        throw(DimensionMismatch("$(m)-dimensional kd-tree searched with a length $(length(x)) vector."))
    end

    for j in 1:N
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

    return _traverse!(bounds, node, x)
end

_traverse!(bounds, node::Nothing, x) = bounds
function _traverse!(bounds, node::KDNode, x)
    if x[node.j] <= node.med
        bounds[2, node.j] = node.med
        return _traverse!(bounds, node.leftnode, x)
    else
        bounds[1, node.j] = node.med
        return _traverse!(bounds, node.rightnode, x)
    end
end
