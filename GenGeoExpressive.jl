using GeometryBasics: Point3
using Graphs

# ----------------------------------------------------------------------------
# 1. Build once: static graph and indexing
# ----------------------------------------------------------------------------
struct GeoGraphReal
    n::Int                     # number of nodes
    flat_x::Vector{Float32}    # x-coords
    flat_y::Vector{Float32}    # y-coords
    neighbors::Vector{Vector{Int}}
    d0::Vector{Vector{Float32}}    # d0[i][k] = flat distance to k-th neighbor
    edges_idx::Vector{Tuple{Int,Int}}
    adjacency::Matrix{Float32}
    # reusable buffers
    points3D::Vector{Point3{Float32}}
    R_vals::Vector{Float32}
    anisotropy::Vector{Float32}
end

function GeoGraphReal(pf_states::Vector{String},
                      flat_pos::Dict{String,Tuple{Float64,Float64}},
                      edges::Vector{Tuple{String,String}})
    n = length(pf_states)
    # string → idx once
    idx_map = Dict(s=>i for (i,s) in enumerate(pf_states))
    # flat coords
    fx = Float32[flat_pos[s][1] for s in pf_states]
    fy = Float32[flat_pos[s][2] for s in pf_states]
    # edge idx list
    eidx = [(idx_map[u], idx_map[v]) for (u,v) in edges]
    # build neighbor index lists
    nbrs = [Int[] for _ in 1:n]
    for (i,j) in eidx
        push!(nbrs[i], j)
        push!(nbrs[j], i)
    end
    # precompute flat distances
    d0 = [ Float32[sqrt((fx[i]-fx[j])^2 + (fy[i]-fy[j])^2) for j in nbrs[i]] 
           for i in 1:n ]
    # adjacency matrix
    g = SimpleGraph(n)
    for (i,j) in eidx
        add_edge!(g,i,j)
    end
    A = Float32.(adjacency_matrix(g))
    # allocate buffers
    pts3D = Vector{Point3{Float32}}(undef, n)
    Rbuf   = zeros(Float32, n)
    anis   = zeros(Float32, n)
    return GeoGraphReal(n, fx, fy, nbrs, d0, eidx, A, pts3D, Rbuf, anis)
end

# ----------------------------------------------------------------------------
# 2. Update per new ρ
# ----------------------------------------------------------------------------
"""
    update_real_geometry!(G::GeoGraphReal, ρ::Vector{Float32})

Fills G.points3D, G.R_vals and G.anisotropy in place.
"""
function update_real_geometry!(G::GeoGraphReal, ρ::Vector{Float32})
    n = G.n
    # 1) lift into 3D buffer
    @inbounds for i in 1:n
        G.points3D[i] = Point3(G.flat_x[i], G.flat_y[i], -ρ[i])
        G.R_vals[i] = 0f0
    end
    # 2) compute R in one pass over edges
    @inbounds for i in 1:n
        pi = G.points3D[i]
        for (k, j) in enumerate(G.neighbors[i])
            pj = G.points3D[j]
            # d0 from cache
            d3 = norm(pi - pj)
            G.R_vals[i] += d3 - G.d0[i][k]
        end
    end
    # 3) anisotropy
    @inbounds for i in 1:n
        acc = 0f0; cnt = 0
        Ri = G.R_vals[i]
        for (k, j) in enumerate(G.neighbors[i])
            dist = G.d0[i][k]
            ΔR = abs(Ri - G.R_vals[j])
            if dist > 1e-6f0
                acc += ΔR/dist
                cnt += 1
            end
        end
        G.anisotropy[i] = cnt>0 ? acc/cnt : 0f0
    end
    return G
end

# ----------------------------------------------------------------------------
# Usage
# ----------------------------------------------------------------------------
pf_states = ["000", "001", "010", "100", "011", "101", "110", "111"]
flat_pos = Dict(
    "000" => (0.0, 3.0), "001" => (-2.0, 2.0), "010" => (0.0, 2.0),
    "100" => (2.0, 2.0), "011" => (-1.0, 1.0), "101" => (0.0, 1.0),
    "110" => (1.0, 1.0), "111" => (0.0, 0.0)
)
edges = [
    ("000", "001"), ("000", "010"), ("000", "100"),
    ("001", "011"), ("001", "101"), ("010", "011"), ("010", "110"),
    ("100", "110"), ("100", "101"), ("011", "111"), ("101", "111"), ("110", "111")
]

# build once
G = GeoGraphReal(pf_states, flat_pos, edges)

# in your loop, for each new ρ:
update_real_geometry!(G, ρ)

# Now G.R_vals and G.anisotropy are up-to-date, and G.adjacency is already built.
