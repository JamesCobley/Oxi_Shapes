# =============================================================================
# GeoBrain: Real-Flow & Imagination Pipeline
# =============================================================================
using LinearAlgebra
using SparseArrays
using GeometryBasics
using Graphs
using StatsBase
using Statistics: mean
using Random
using UUIDs
using BSON: @save, @load
using Dates

# =============================================================================
# Graph Manifold Construction
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
# Real-Flow (Oxi-Shapes Alive)
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
# Imagination Pipeline
# =============================================================================
struct LivingSimplexTensor
    real::Vector{Float32}
    imag::Vector{Float32}
end

init_living_simplex_tensor(ρ0::Vector{Float32}) =
    LivingSimplexTensor(copy(ρ0), copy(ρ0))

build_imagined_manifold!(field::LivingSimplexTensor, G::GeoGraphReal) =
    update_real_geometry!(G, field.imag)

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

compute_lambda(ρ_real::Vector{Float32}, ρ_imag::Vector{Float32}) =
    mean(abs.(ρ_imag .- ρ_real))

function step_imagination!(field::LivingSimplexTensor,
                           G::GeoGraphReal,
                           buffers;
                           max_moves::Int=10)
    build_imagined_manifold!(field,G)
    evolve_imagination_single_counts!(field,G,buffers; moves=max_moves)
    return compute_lambda(field.real, field.imag)
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

# =============================================================================
# 3. GeoNode & Simplex Storage (Updated for primitives)
# =============================================================================
struct GeoNode
    ρ_real::Vector{Float32}
    ρ_imag::Vector{Float32}
    R_vals::Vector{Float32}
    anisotropy::Vector{Float32}
    λ::Float32
    flux::Vector{Float32}         # primitive flow vector (ρ_new - ρ_old)
    action_cost::Float32          # primitive action cost (sum(abs(flux)))
end

init_simplex(n_runs::Int) = [GeoNode[] for _ in 1:n_runs]

function record_geonode!(simplex, run_idx, field, ρ_old::Vector{Float32}, ρ_new::Vector{Float32}, G::GeoGraphReal, buffers;
                         max_moves_imag::Int=10)
    # 1) Compute flux & action cost from real-flow
    flux_vec    = ρ_new .- ρ_old
    action_cost = sum(abs.(flux_vec))

    # 2) Imagination update
    field.real .= ρ_new
    λ = step_imagination!(field, G, buffers; max_moves=max_moves_imag)

    # 3) Capture updated geometry
    R_im   = copy(G.R_vals)
    a_im   = copy(G.anisotropy)

    # 4) Build GeoNode with primitives
    node = GeoNode(
      copy(ρ_new),
      copy(field.imag),
      R_im,
      a_im,
      λ,
      copy(flux_vec),
      action_cost
    )

    # 5) Store
    push!(simplex[run_idx], node)
    return node
end

# =============================================================================
# Meta-Lambda as an Adaptive Hypergraph Controller (with primitives)
# =============================================================================

mutable struct MetaLambdaHypergraph
    run_w            :: Vector{Float32}  # one weight per run-edge
    curvature_w      :: Vector{Float32}  # one per curvature bin
    action_cost_w    :: Vector{Float32}  # one per action_cost bin
    transition_w     :: Vector{Float32}  # one per transition pattern
end

function init_meta_hypergraph(hyperedges::Dict)
    return MetaLambdaHypergraph(
        ones(Float32, length(hyperedges[:run])),
        ones(Float32, length(hyperedges[:curvature])),
        ones(Float32, length(hyperedges[:action_cost])),
        ones(Float32, length(hyperedges[:transition_pattern]))
    )
end

function update_meta_hypergraph!(
    ml::MetaLambdaHypergraph,
    hyperedges::Dict{Symbol, Vector{Vector{Int}}},
    nodes::Vector{GeoNode},
    η::Float32;
    λ_target::Float32 = 0.05f0
)
    # run edges
    for (i, e) in enumerate(hyperedges[:run])
        err = mean(getfield.(nodes[e], :λ)) - λ_target
        ml.run_w[i] -= η * err
    end

    # curvature edges
    for (i, e) in enumerate(hyperedges[:curvature])
        err = mean(getfield.(nodes[e], :λ)) - λ_target
        ml.curvature_w[i] -= η * err
    end

    # action_cost edges
    for (i, e) in enumerate(hyperedges[:action_cost])
        err = mean(getfield.(nodes[e], :λ)) - λ_target
        ml.action_cost_w[i] -= η * err
    end

    # transition_pattern edges
    for (i, e) in enumerate(hyperedges[:transition_pattern])
        err = mean(getfield.(nodes[e], :λ)) - λ_target
        ml.transition_w[i] -= η * err
    end
end

# =============================================================================
# MetaController with Scalar Information Potential
# =============================================================================

mutable struct MetaController
    run_w            :: Vector{Float32}
    curvature_w      :: Vector{Float32}
    action_cost_w    :: Vector{Float32}
    transition_w     :: Vector{Float32}
    info_potential   :: Vector{Float32}   # scalar φ per node
end

function init_meta_controller(hyperedges::Dict, G::GeoGraphReal)
    nr   = length(hyperedges[:run])
    nc   = length(hyperedges[:curvature])
    na   = length(hyperedges[:action_cost])
    nt   = length(hyperedges[:transition_pattern])

    φ = zeros(Float32, G.n)

    return MetaController(
        ones(Float32, nr),
        ones(Float32, nc),
        ones(Float32, na),
        ones(Float32, nt),
        φ
    )
end

# =============================================================================
# step_imagination! → use information potential gradient
# =============================================================================

function step_imagination!(
    field::LivingSimplexTensor,
    G::GeoGraphReal,
    buffers,
    node_idx::Int,
    hyperedges::Dict{Symbol, Vector{Vector{Int}}},
    ml::MetaController;
    max_moves::Int = 10
)
    # 1) rebuild imagined geometry
    build_imagined_manifold!(field, G)

    # 2) compute hyperedge bias as before
    run_e   = findfirst(i-> node_idx in hyperedges[:run][i], 1:length(hyperedges[:run]))
    curv_e  = findfirst(i-> node_idx in hyperedges[:curvature][i], 1:length(hyperedges[:curvature]))
    cost_e  = findfirst(i-> node_idx in hyperedges[:action_cost][i], 1:length(hyperedges[:action_cost]))
    trans_e = findfirst(i-> node_idx in hyperedges[:transition_pattern][i], 1:length(hyperedges[:transition_pattern]))

    hb = ml.run_w[run_e] +
         ml.curvature_w[curv_e] +
         ml.action_cost_w[cost_e] +
         ml.transition_w[trans_e]

    # 3) perform discrete moves with φ-gradient bias
    counts = buffers.counts
    fill!(buffers.inflow_int,0); fill!(buffers.outflow_int,0)
    total_moves = rand(0:max_moves)

    for _ in 1:total_moves
        nonzero = findall(>(0), counts)
        isempty(nonzero)&& break
        i = rand(nonzero)
        nbrs = G.neighbors[i]
        isempty(nbrs)&& continue

        # compute weights including discrete gradient bias
        ws = Float32[]
        for (k,j) in enumerate(nbrs)
            ΔS = 0.1f0 + 0.01f0 + abs(G.R_vals[j]) + buffers.deg_pen[j]
            Δf = exp(counts[i]/100) - G.R_vals[i] + ΔS
            base = exp(-Δf) * exp(-hb)
            # gradient of φ
            grad = ml.info_potential[j] - ml.info_potential[i]
            push!(ws, base * exp(grad))
        end

        # normalize and sample
        wsum = sum(ws)
        wsum<1e-8f0 && continue
        r=rand()*wsum; cum=0f0; chosen=nbrs[1]
        for (k,j) in enumerate(nbrs)
            cum+=ws[k]
            if cum>=r; chosen=j; break; end
        end
        buffers.inflow_int[chosen]+=1
        buffers.outflow_int[i]+=1
    end

    # 4) apply flows & update imag
    @inbounds for i in 1:G.n
        counts[i]+=buffers.inflow_int[i]-buffers.outflow_int[i]
        field.imag[i]=counts[i]/100f0
    end

    # 5) return divergence
    return compute_lambda(field.real, field.imag)
end

# =============================================================================
# Update the Information Potential from λ-errors
# =============================================================================

function update_info_potential!(
    ml::MetaController,
    all_nodes::Vector{GeoNode},
    η::Float32;
    λ_target::Float32 = 0.05f0
)
    # accumulate error sums and counts
    err_sum = zeros(Float32, length(ml.info_potential))
    cnt     = zeros(Int, length(err_sum))
    for node in all_nodes
        i = node.node_index
        err_sum[i] += (λ_target - node.λ)
        cnt[i] += 1
    end
    # update φ
    for i in eachindex(err_sum)
        if cnt[i] > 0
            Δφ = err_sum[i] / cnt[i]
            ml.info_potential[i] += η * Δφ
        end
    end
    return ml
end
struct GeoNode
    ρ_real::Vector{Float32}
    ρ_imag::Vector{Float32}
    R_vals::Vector{Float32}
    anisotropy::Vector{Float32}
    λ::Float32
    flux::Vector{Float32}         # primitive flow vector (ρ_new - ρ_old)
    action_cost::Float32          # primitive action cost (sum(abs(flux)))
end

init_simplex(n_runs::Int) = [GeoNode[] for _ in 1:n_runs]

function record_geonode!(simplex, run_idx, field, ρ_old::Vector{Float32}, ρ_new::Vector{Float32}, G::GeoGraphReal, buffers;
                         max_moves_imag::Int=10)
    # 1) Compute flux & action cost from real-flow
    flux_vec    = ρ_new .- ρ_old
    action_cost = sum(abs.(flux_vec))

    # 2) Imagination update
    field.real .= ρ_new
    λ = step_imagination!(field, G, buffers; max_moves=max_moves_imag)

    # 3) Capture updated geometry
    R_im   = copy(G.R_vals)
    a_im   = copy(G.anisotropy)

    # 4) Build GeoNode with primitives
    node = GeoNode(
      copy(ρ_new),
      copy(field.imag),
      R_im,
      a_im,
      λ,
      copy(flux_vec),
      action_cost
    )

    # 5) Store
    push!(simplex[run_idx], node)
    return node
end

# =============================================================================
# Hypergraph Construction with Primitives
# =============================================================================
function build_hypergraph(simplex; curvature_bins::Int=5, cost_bins::Int=5)
    nodes = reduce(vcat, simplex)
    N = length(nodes)

    # 1) run hyperedges
    run_edges = Vector{Vector{Int}}()
    idx = 1
    for run_vec in simplex
        push!(run_edges, collect(idx:idx+length(run_vec)-1))
        idx += length(run_vec)
    end

    # 2) curvature hyperedges
    Rmean = [mean(n.R_vals) for n in nodes]
    bins_c = StatsBase.cut(Rmean, curvature_bins)
    curv_groups = Dict{Int,Vector{Int}}()
    for (i,b) in enumerate(bins_c)
        push!(get!(curv_groups,b,Int[]), i)
    end
    curv_edges = collect(values(curv_groups))

    # 3) action_cost hyperedges
    costs = [n.action_cost for n in nodes]
    bins_cost = StatsBase.cut(costs, cost_bins)
    cost_groups = Dict{Int,Vector{Int}}()
    for (i,b) in enumerate(bins_cost)
        push!(get!(cost_groups,b,Int[]), i)
    end
    cost_edges = collect(values(cost_groups))

    # 4) transition_pattern hyperedges
    trans_map = Dict{Vector{Int},Vector{Int}}()
    for (i,n) in enumerate(nodes)
        moves = findall(!=(0f0), n.flux)
        push!(get!(trans_map,moves,Int[]), i)
    end
    trans_edges = collect(values(trans_map))

    return Dict(
      :run                => run_edges,
      :curvature          => curv_edges,
      :action_cost        => cost_edges,
      :transition_pattern => trans_edges
    ), nodes
end

    # 2) curvature hyperedges
    Rmean = [mean(n.R_vals) for n in nodes]
    bins_c = StatsBase.cut(Rmean, curvature_bins)
    curv_groups = Dict{Int,Vector{Int}}()
    for (i,b) in enumerate(bins_c)
        push!(get!(curv_groups,b,Int[]), i)
    end
    curv_edges = collect(values(curv_groups))

    # 3) action_cost hyperedges
    costs = [n.action_cost for n in nodes]
    bins_cost = StatsBase.cut(costs, cost_bins)
    cost_groups = Dict{Int,Vector{Int}}()
    for (i,b) in enumerate(bins_cost)
        push!(get!(cost_groups,b,Int[]), i)
    end
    cost_edges = collect(values(cost_groups))

    # 4) transition_pattern hyperedges
    trans_map = Dict{Vector{Int},Vector{Int}}()
    for (i,n) in enumerate(nodes)
        moves = findall(!=(0f0), n.flux)
        push!(get!(trans_map,moves,Int[]), i)
    end
    trans_edges = collect(values(trans_map))

    return Dict(
      :run                => run_edges,
      :curvature          => curv_edges,
      :action_cost        => cost_edges,
      :transition_pattern => trans_edges
    ), nodes
end
struct GeoNode
    ρ_real::Vector{Float32}
    ρ_imag::Vector{Float32}
    R_vals::Vector{Float32}
    anisotropy::Vector{Float32}
    λ::Float32
    flux::Vector{Float32}         # primitive flow vector (ρ_new - ρ_old)
    action_cost::Float32          # primitive action cost (sum(abs(flux)))
end

function generate_safe_random_initials(n::Int, R::Int; total::Int = 100)
    batch_id = "batch_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
    samples = Vector{NamedTuple{(:uuid,:counts,:rho),Tuple{UUID,Vector{Int},Vector{Float32}}}}()

    for _ in 1:n
        # 1) Draw R-1 cut points between 0 and total
        cuts = sort(rand(0:total, R-1))
        # 2) Form counts by taking differences
        counts = Vector{Int}(undef, R)
        counts[1] = cuts[1]
        for i in 2:R-1
            counts[i] = cuts[i] - cuts[i-1]
        end
        counts[R] = total - cuts[end]

        # 3) Normalize for ρ
        rho = Float32.(counts) ./ Float32(total)

        push!(samples, (uuid=uuid4(), counts=counts, rho=rho))
    end

    return batch_id, samples
end

    train_geobrain!(
      pf_states, flat_pos, edges, bitcounts;
      n_runs::Int=100,
      rollout_steps::Int=50,
      max_moves_real::Int=10,
      max_moves_imag::Int=10,
      curvature_bins::Int=5,
      cost_bins::Int=5,
      η_hyper::Float32=1e-2,
      η_potential::Float32=1e-2,
      λ_target::Float32=0.05f0,
      max_epochs::Int=100
    ) -> (ml_ctrl::MetaController, hyperedges, all_nodes)

function train_geobrain!(
    pf_states::Vector{String},
    flat_pos::Dict{String,Tuple{Float64,Float64}},
    edges::Vector{Tuple{String,String}},
    bitcounts::Vector{Int};
    n_runs::Int=100,
    rollout_steps::Int=50,
    max_moves_real::Int=10,
    max_moves_imag::Int=10,
    curvature_bins::Int=5,
    cost_bins::Int=5,
    η_hyper::Float32=1e-2,
    η_potential::Float32=1e-2,
    λ_target::Float32=0.05f0,
    max_epochs::Int=100
)
    R = length(bitcounts)
    # 0) Static initializations
    G       = GeoGraphReal(pf_states, flat_pos, edges)
    buffers = init_alive_buffers!(G, bitcounts)
    # pre‐allocate imagination field (we'll sync real each run)
    field   = init_living_simplex_tensor(zeros(Float32, R))
    # storage
    simplex = init_simplex(n_runs)

    # controller placeholders (we'll init after first hypergraph build)
    ml_ctrl = nothing
    hyperedges = nothing
    all_nodes = Vector{GeoNode}()

    for epoch in 1:max_epochs
        println("=== Epoch $epoch ===")

        # 1) generate fresh initials
        _, samples = generate_safe_random_initials(n_runs, R; total=100)
        initials = [s.rho for s in samples]

        # clear previous simplex
        simplex = init_simplex(n_runs)

        # 2) for each run, do real‐flow + record GeoNodes
        for run_idx in 1:n_runs
            ρ = copy(initials[run_idx])
            # seed imag = real
            field.real .= ρ
            field.imag .= ρ

            for step in 1:rollout_steps
                ρ_old = copy(ρ)
                # real‐flow update
                oxi_shapes_alive!(ρ, G, buffers; max_moves=max_moves_real)
                # record real→imag
                record_geonode!(simplex, run_idx, field, ρ_old, ρ, G, buffers;
                                max_moves_imag=max_moves_imag)
            end
        end

        # 3) build hypergraph on all newly recorded nodes
        hyperedges, all_nodes = build_hypergraph(simplex;
                                                 curvature_bins=curvature_bins,
                                                 cost_bins=cost_bins)

        # 4) initialize controller on first epoch
        if epoch == 1
            ml_ctrl = init_meta_controller(hyperedges, G)
        end

        # 5) update hyperedge‐weights & info_potential
        update_meta_hypergraph!(ml_ctrl, hyperedges, all_nodes, η_hyper;
                                λ_target=λ_target)
        update_info_potential!(ml_ctrl, all_nodes, η_potential;
                               λ_target=λ_target)

        # 6) convergence check
        avg_λ = mean(n.λ for n in all_nodes)
        println(" → avg λ = $(round(avg_λ, digits=4))  (target = $λ_target)")
        if avg_λ ≤ λ_target
            println("Converged after $epoch epochs!")
            break
        end
    end

    return ml_ctrl, hyperedges, all_nodes
end

function save_geobrain(path::AbstractString,
                       ml_ctrl::MetaController,
                       hyperedges::Dict{Symbol,Vector{Vector{Int}}},
                       all_nodes::Vector{GeoNode},
                       run_log::Vector{Dict},
                       simplex::Vector{Vector{GeoNode}})
    @save path ml_ctrl hyperedges all_nodes run_log simplex
    println("GeoBrain saved to ", path)
end

# — example usage, immediately after train_geobrain! returns —
ml_ctrl, hyperedges, all_nodes = train_geobrain!(...; max_epochs=50)

# Suppose you collected a `run_log` during training, e.g.:
# run_log = [ Dict(:epoch=>1, :avg_λ=>0.12), Dict(:epoch=>2, :avg_λ=>0.08), … ]

# And you kept the last simplex:
# simplex :: Vector{Vector{GeoNode}}

save_geobrain("geobrain_model_$(Dates.format(now(), "yyyymmdd_HHMMSS")).bson",
              ml_ctrl,
              hyperedges,
              all_nodes,
              run_log,
              simplex)
