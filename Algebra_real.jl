using LinearAlgebra, SparseArrays
using BSON: @load

# -------------------- Config --------------------
trace_path = "/content/trace_metadata_batch_20250524_152552.bson"

const STATES = ["000","001","010","011","100","101","110","111"]
const IDX    = Dict(s=>i for (i,s) in enumerate(STATES))
const R = length(first(STATES))
const N = length(STATES)

# -------------------- Helpers --------------------
bit_index(s::String, t::String) = begin
    @assert length(s) == length(t)
    k = 0
    @inbounds for i in 1:length(s)
        if s[i] != t[i]
            k == 0 || return 0
            k = i
        end
    end
    return k
end

# Collect transitions (stepwise flips)
function collect_transitions(trace_metadata)
    trips = Tuple{Int,Int,Int}[]
    for meta in trace_metadata
        raw = haskey(meta, :compressed) ? meta[:compressed] :
              haskey(meta, :steps)      ? meta[:steps]      : String[]
        steps = String.(raw)
        steps = [s for s in steps if haskey(IDX, s)]
        for i in 1:(length(steps)-1)
            s, t = steps[i], steps[i+1]
            b = bit_index(s,t)
            b == 0 && continue
            push!(trips, (IDX[t], IDX[s], b))
        end
    end
    return trips
end

# Build empirical operators M_i
function build_generators(trips; smoothing=1e-9)
    Gs = [spzeros(Float64, N, N) for _ in 1:R]
    for (to, from, b) in trips
        Gs[b][to, from] += 1.0
    end
    # add smoothing and normalize
    for b in 1:R
        G = Gs[b]
        for j in 1:N
            s = sum(@view G[:,j])
            if s > 0
                @views G[:,j] ./= s
            else
                G[j,j] = 1.0 # identity if unseen
            end
        end
    end
    return Gs
end

# Commutator norms
function commutator_norms(Gs)
    B = length(Gs)
    Cnorm = zeros(Float64, B, B)
    for i in 1:B, j in 1:B
        C = Gs[i]*Gs[j] - Gs[j]*Gs[i]
        Cnorm[i,j] = norm(Matrix(C))
    end
    return Cnorm
end

# Path-level algebra: order counts
function path_noncommutativity(trace_metadata)
    counts = Dict{Tuple{Int,Int},Int}()
    for meta in trace_metadata
        raw = haskey(meta, :compressed) ? meta[:compressed] :
              haskey(meta, :steps)      ? meta[:steps]      : String[]
        steps = String.(raw)
        for i in 1:(length(steps)-2)
            b1 = bit_index(steps[i], steps[i+1])
            b2 = bit_index(steps[i+1], steps[i+2])
            if b1 > 0 && b2 > 0 && b1 != b2
                counts[(b1,b2)] = get(counts,(b1,b2),0)+1
            end
        end
    end
    return counts
end

# -------------------- Run --------------------
@load trace_path trace_metadata
trace_metadata = Dict{Symbol,Any}.(trace_metadata)

# Operators
trips = collect_transitions(trace_metadata)
Gs    = build_generators(trips)
C     = commutator_norms(Gs)

println("=== Operator Algebra (commutator norms) ===")
println(C)

println("\n=== Path Algebra (motif orderings) ===")
counts = path_noncommutativity(trace_metadata)
for ((i,j),c) in counts
    println("M$i → M$j : $c")
end
for i in 1:R, j in i+1:R
    c_ij = get(counts,(i,j),0)
    c_ji = get(counts,(j,i),0)
    if c_ij != c_ji
        println("Non-commuting pair (M$i,M$j): $c_ij vs $c_ji")
    end
end
