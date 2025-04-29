# =============================================================================
# GeoBrain: Real-Flow & Imagination Pipeline
# =============================================================================
using GeometryBasics: Point3
using Graphs
using Statistics: mean
using Random

# =============================================================================
# 0. Graph Manifold Construction
# =============================================================================
struct GeoGraphReal
    n::Int
    flat_x::Vector{Float32}
    flat_y::Vector{Float32}
    neighbors::Vector{Vector{Int}}
    d0::Vector{Vector{Float32}}
    edges_idx::Vector{Tuple{Int,Int}}
    adjacency::Matrix{Float32}
    points3D::Vector{Point3{Float32}}
    R_vals::Vector{Float32}
    anisotropy::Vector{Float32}
end

function GeoGraphReal(pf_states::Vector{String},
                      flat_pos::Dict{String,Tuple{Float64,Float64}},
                      edges::Vector{Tuple{String,String}})
    n = length(pf_states)
    idx_map = Dict(s=>i for (i,s) in enumerate(pf_states))
    fx = Float32[flat_pos[s][1] for s in pf_states]
    fy = Float32[flat_pos[s][2] for s in pf_states]
    eidx = [(idx_map[u], idx_map[v]) for (u,v) in edges]
    nbrs = [Int[] for _ in 1:n]
    for (i,j) in eidx
        push!(nbrs[i], j)
        push!(nbrs[j], i)
    end
    d0 = [Float32[sqrt((fx[i]-fx[j])^2 + (fy[i]-fy[j])^2) for j in nbrs[i]] for i in 1:n]
    g = SimpleGraph(n)
    for (i,j) in eidx
        add_edge!(g,i,j)
    end
    A = Float32.(adjacency_matrix(g))
    pts3D = Vector{Point3{Float32}}(undef, n)
    Rbuf  = zeros(Float32, n)
    anis  = zeros(Float32, n)
    return GeoGraphReal(n, fx, fy, nbrs, d0, eidx, A, pts3D, Rbuf, anis)
end

"""
update_real_geometry!(G, ρ)
Lift to 3D and compute curvature + anisotropy in-place.
"""
function update_real_geometry!(G::GeoGraphReal, ρ::Vector{Float32})
    @inbounds for i in 1:G.n
        G.points3D[i] = Point3(G.flat_x[i], G.flat_y[i], -ρ[i])
        G.R_vals[i] = 0f0
    end
    @inbounds for i in 1:G.n
        pi = G.points3D[i]
        for (k,j) in enumerate(G.neighbors[i])
            d3 = norm(pi - G.points3D[j])
            G.R_vals[i] += d3 - G.d0[i][k]
        end
    end
    @inbounds for i in 1:G.n
        acc, cnt = 0f0, 0
        Ri = G.R_vals[i]
        for (k,j) in enumerate(G.neighbors[i])
            dist = G.d0[i][k]
            ΔR = abs(Ri - G.R_vals[j])
            if dist > 1e-6f0
                acc += ΔR/dist; cnt += 1
            end
        end
        G.anisotropy[i] = cnt>0 ? acc/cnt : 0f0
    end
    return G
end

# =============================================================================
# 1. Real-Flow (Oxi-Shapes Alive)
# =============================================================================
"""
init_alive_buffers!(G, bitcounts)
Pre-allocate counts, inflow, outflow, and degeneracy penalties.
"""
function init_alive_buffers!(G::GeoGraphReal, bitcounts::Vector{Int})
    n = G.n
    counts = Vector{Int}(undef, n)
    inflow_int = zeros(Int, n)
    outflow_int= zeros(Int, n)
    R_total = length(bitcounts)
    binom = Dict(k=>binomial(R_total,k) for k in 0:R_total)
    deg_pen = Float32[1f0/binom[bitcounts[i]] for i in 1:n]
    return (counts=counts, inflow_int=inflow_int,
            outflow_int=outflow_int, deg_pen=deg_pen)
end

"""
oxi_shapes_alive!(ρ, G, buffers; max_moves)
Discrete-count stochastic flow; updates ρ in-place.
"""
function oxi_shapes_alive!(ρ::Vector{Float32}, G::GeoGraphReal, buffers;
                           max_moves::Int=10)
    n = G.n; counts = buffers.counts
    @inbounds for i in 1:n counts[i] = round(Int,ρ[i]*100) end
    counts[n] = 100 - sum(counts[1:n-1])
    update_real_geometry!(G, ρ)
    fill!(buffers.inflow_int,0); fill!(buffers.outflow_int,0)
    total_moves = rand(0:max_moves)
    nonzero = findall(>(0),counts)
    for _ in 1:total_moves
        isempty(nonzero)&&break
        i = rand(nonzero); nbrs = G.neighbors[i]
        isempty(nbrs)&&continue
        wsum=0f0; ws = Float32[]; push!(ws,0f0) # scratch
        for j in nbrs
            ΔS = 0.1f0+0.01f0+abs(G.R_vals[j])+buffers.deg_pen[j]
            Δf = exp(counts[i]/100)-G.R_vals[i]+ΔS
            w = exp(-Δf)*exp(-G.anisotropy[j]); wsum+=w; push!(ws,w)
        end
        wsum<1e-8f0 && continue
        r=rand()*wsum; cum=0f0; chosen=nbrs[1]
        for (k,j) in enumerate(nbrs)
            cum+=ws[k+1]; if cum>=r chosen=j; break end
        end
        buffers.inflow_int[chosen]+=1; buffers.outflow_int[i]+=1
        if counts[i]-buffers.outflow_int[i]==0
            deleteat!(nonzero, findfirst(==(i),nonzero))
        end
        if counts[chosen]+buffers.inflow_int[chosen]==1
            push!(nonzero, chosen)
        end
    end
    @inbounds for i in 1:n
        counts[i]+=buffers.inflow_int[i]-buffers.outflow_int[i]
        ρ[i]=counts[i]/100f0
    end
    return ρ
end

# =============================================================================
# 2. Imagination Pipeline
# =============================================================================
struct LivingSimplexTensor
    real::Vector{Float32}
    imag::Vector{Float32}
end

"""
init_living_simplex_tensor(ρ0)
Initialize real & imag fields.
"""
init_living_simplex_tensor(ρ0::Vector{Float32}) =
    LivingSimplexTensor(copy(ρ0), copy(ρ0))

"""
build_imagined_manifold!(field,G)
Update G geometry based on imag.
"""
build_imagined_manifold!(field::LivingSimplexTensor, G::GeoGraphReal) =
    update_real_geometry!(G, field.imag)

"""
evolve_imagination_single_counts!(field,G,buffers;moves)
Deterministic moves on imag counts.
"""
function evolve_imagination_single_counts!(field::LivingSimplexTensor,
                                         G::GeoGraphReal,
                                         buffers;
                                         moves::Int=10)
    counts=buffers.counts
    @inbounds for i in 1:G.n counts[i]=round(Int,field.imag[i]*100) end
    counts[G.n]=100-sum(counts[1:G.n-1])
    update_real_geometry!(G,field.imag)
    fill!(buffers.inflow_int,0); fill!(buffers.outflow_int,0)
    nmoves=rand(0:moves); nonzero=findall(>(0),counts)
    for _ in 1:nmoves
        isempty(nonzero)&&break
        i=nonzero[argmax(G.R_vals[nonzero])]; nbrs=G.neighbors[i]
        isempty(nbrs)&&continue
        best_j,bc= nbrs[1],Inf
        for j in nbrs
            ΔS=0.1f0+0.01f0+abs(G.R_vals[j])+buffers.deg_pen[j]
            Δf=exp(counts[i]/100)-G.R_vals[i]+ΔS
            if Δf<bc bc, best_j=Δf,j end
        end
        buffers.inflow_int[best_j]+=1; buffers.outflow_int[i]+=1
        if counts[i]-buffers.outflow_int[i]==0
            deleteat!(nonzero, findfirst(==(i),nonzero))
        end
        if counts[best_j]+buffers.inflow_int[best_j]==1
            push!(nonzero,best_j)
        end
    end
    @inbounds for i in 1:G.n
        counts[i]+=buffers.inflow_int[i]-buffers.outflow_int[i]
        field.imag[i]=counts[i]/100f0
    end
    return field.imag
end

"""
compute_lambda(r,i)
Mean absolute difference.
"""
compute_lambda(ρ_real::Vector{Float32}, ρ_imag::Vector{Float32}) =
    mean(abs.(ρ_imag .- ρ_real))

"""
step_imagination!(field,G,buffers)
One imag step, returns λ.
"""
function step_imagination!(field::LivingSimplexTensor,
                           G::GeoGraphReal,
                           buffers;
                           max_moves::Int=10)
    build_imagined_manifold!(field,G)
    evolve_imagination_single_counts!(field,G,buffers; moves=max_moves)
    return compute_lambda(field.real, field.imag)
end

# =============================================================================
# 3. GeoNode & Simplex Storage
# =============================================================================
struct GeoNode
    ρ_real::Vector{Float32}
    ρ_imag::Vector{Float32}
    R_vals::Vector{Float32}
    anisotropy::Vector{Float32}
    λ::Float32
end

init_simplex(n_runs::Int) = [GeoNode[] for _ in 1:n_runs]

"""
record_geonode!(simplex,run,field,ρ,G,buffers)
Wrap and store one GeoNode.
"""
function record_geonode!(simplex, run_idx, field, ρ_real, G, buffers;
                         max_moves_imag::Int=10)
    field.real .= ρ_real
    λ = step_imagination!(field,G,buffers; max_moves=max_moves_imag)
    R_im = copy(G.R_vals); a_im = copy(G.anisotropy)
    node = GeoNode(copy(ρ_real), copy(field.imag), R_im, a_im, λ)
    push!(simplex[run_idx], node)
    return node
end

# =============================================================================
# 4. Flow-Helper: time-series metrics from GeoNode runs
# =============================================================================

# --- 1) FlowMetrics container
struct FlowMetrics
    ρ_series::Vector{Vector{Float32}}
    flux_series::Vector{Vector{Float32}}
    R_series::Vector{Vector{Float32}}
    anis_series::Vector{Vector{Float32}}
    k_series::Vector{Dict{Int,Float32}}
    mean_ox::Vector{Float32}
    entropy::Vector{Float32}
    fisher::Vector{Float32}
    transition::Vector{Symbol}
    on_geodesic::Vector{Bool}
    geodesic_path::Vector{String}
end

# --- 2) Compute k-distribution
function compute_k_distribution(ρ::Vector{Float32}, pf_states::Vector{String})
    kdist = Dict{Int,Float32}()
    for (i,s) in enumerate(pf_states)
        k = count(==('1'), s)
        kdist[k] = get(kdist,k,0f0) + ρ[i]
    end
    return kdist
end

# --- 3) Shannon entropy & Fisher info
shannon_entropy(p) = -sum(x-> x>0f0 ? x*log2(x) : 0f0, p)
function fisher_information(ρ::Vector{Float32})
    g = diff(ρ)
    return sum(g .^ 2)
end

# --- 4) Geodesic classification
function dominant_geodesic(traj::Vector{String}, geodesics)
    best,ms = nothing,0
    for path in geodesics
        c = count(in(path), traj)
        if c>ms; ms, best = c, path; end
    end
    return best
end

function classify_transitions(ρs::Vector{Vector{Float32}}, pf_states)
    n = length(ρs)
    classes = Symbol[]
    for t in 2:n
        prev, curr = ρs[t-1], ρs[t]
        kprev = sum(count(==('1'), pf_states[i])*prev[i] for i in 1:length(prev))/length(pf_states)
        kcurr = sum(count(==('1'), pf_states[i])*curr[i] for i in 1:length(curr))/length(pf_states)
        push!(classes,
            kcurr >  kprev+1e-4 ? :oxidizing  :
            kcurr <  kprev-1e-4 ? :reducing   :
                                 :neutral)
    end
    return classes
end

# --- 5) On-geodesic flags
function on_geodesic_flags(traj::Vector{String}, path::Vector{String})
    return [s in path for s in traj]
end

# --- 6) Master flow-metrics builder
"""
    build_flow_metrics(simplex_run, pf_states; geodesics)

Given a vector of GeoNode for one run, return FlowMetrics.
"""
function build_flow_metrics(
    run_nodes::Vector{GeoNode},
    pf_states::Vector{String};
    geodesics
)
    T = length(run_nodes)
    # extract series
    ρs   = [node.ρ_real       for node in run_nodes]
    flux = [run_nodes[t].ρ_real .- run_nodes[t-1].ρ_real for t in 2:T]
    Rs   = [node.R_vals        for node in run_nodes]
    As   = [node.anisotropy    for node in run_nodes]
    ks   = [compute_k_distribution(ρs[t], pf_states) for t in 1:T]
    # simple scalars
    mean_ox = [ sum(Int(k)*v for (k,v) in ks[t]) for t in 1:T ]
    entropy = [ shannon_entropy(ρs[t])     for t in 1:T ]
    fisher  = [ fisher_information(ρs[t])  for t in 1:T ]
    # trajectory & geodesic
    traj     = [ pf_states[argmax(node.ρ_real)] for node in run_nodes ]
    geo_path = dominant_geodesic(traj, geodesics)
    on_geo   = on_geodesic_flags(traj, geo_path)
    # transition classes
    trans = classify_transitions(ρs, pf_states)

    return FlowMetrics(
      ρ_series        = ρs,
      flux_series     = flux,
      R_series        = Rs,
      anis_series     = As,
      k_series        = ks,
      mean_ox         = Float32.(mean_ox),
      entropy         = Float32.(entropy),
      fisher          = Float32.(fisher),
      transition      = trans,
      on_geodesic     = on_geo,
      geodesic_path   = geo_path
    )
end

