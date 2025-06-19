using StaticArrays, LinearAlgebra, Graphs

# =============================================================================
# 1️⃣ Build modal graph and compute Laplacian eigenmodes
# =============================================================================

pf_states = ["000","001","010","100","011","101","110","111"]
flat_pos = Dict(
    "000"=>(0.0,3.0), "001"=>(-2.0,2.0), "010"=>(0.0,2.0), "100"=>(2.0,2.0),
    "011"=>(-1.0,1.0), "101"=>(0.0,1.0), "110"=>(1.0,1.0), "111"=>(0.0,0.0)
)
edges = [
    ("000","001"),("000","010"),("000","100"),("001","011"),
    ("001","101"),("010","011"),("010","110"),("100","110"),
    ("100","101"),("011","111"),("101","111"),("110","111")
]

struct GeoGraphReal2
    pf_states::Vector{String}
    neighbors::Vector{Vector{Int}}
    adjacency::Matrix{Float64}
    R_vals::Vector{Float64}
    anisotropy::Vector{SVector{3,Float64}}
end

function GeoGraphReal2(states, edges)
    n = length(states)
    idx = Dict(s=>i for (i,s) in enumerate(states))
    nbr = [Int[] for _ in 1:n]
    A = zeros(Float64, n, n)
    for (u,v) in edges
        i,j = idx[u], idx[v]
        push!(nbr[i],j); push!(nbr[j],i)
        A[i,j] = 1; A[j,i] = 1
    end
    return GeoGraphReal2(states, nbr, A, zeros(n), [SVector(0.,0.,0.) for _=1:n])
end

G = GeoGraphReal2(pf_states, edges)

deg = sum(G.adjacency, dims=2)[:]
L = Diagonal(deg) - G.adjacency
λ, V = eigen(Symmetric(L))

# =============================================================================
# 2️⃣ Ricci curvature & anisotropy update function
# =============================================================================

function update_geometry!(G::GeoGraphReal2, rho::Vector{Float64}, coords, cys)
    fill!(G.R_vals, 0.0)
    for i in 1:length(G.R_vals), j in G.neighbors[i]
        G.R_vals[i] += rho[j] - rho[i]
    end

    for i in 1:length(G.R_vals)
        bits = BitVector(c == '1' for c in collect(G.pf_states[i]))
        on_idx = findall(bits)
        if isempty(on_idx)
            G.anisotropy[i] = SVector(0., 0., 0.)
            continue
        end
        # Safely get 3D coordinates of active cysteines
        pos = []
        for k in on_idx
            if k <= length(cys) && cys[k] <= length(coords)
                push!(pos, coords[cys[k]])
            end
        end
        if isempty(pos)
            G.anisotropy[i] = SVector(0., 0., 0.)
            continue
        end

        com = sum(pos) / length(pos)
        disp = [p - com for p in pos]
        G.anisotropy[i] = sum(disp) / length(disp)

        # Optional debug
        # println("State $(G.pf_states[i]) → bits on: $on_idx → cys: ", [cys[k] for k in on_idx])
    end
end


# =============================================================================
# 3️⃣ Real-space shape handling
# =============================================================================

function load_ca_and_cys(pdb_path::String)
    coords, resnams = SVector{3,Float64}[], String[]
    open(pdb_path, "r") do io
        for ln in eachline(io)
            if startswith(ln, "ATOM")
                atom = strip(ln[13:16]); res = strip(ln[18:20])
                if atom == "CA"
                    x = parse(Float64, ln[31:38])
                    y = parse(Float64, ln[39:46])
                    z = parse(Float64, ln[47:54])
                    push!(coords, SVector(x,y,z))
                    push!(resnams, res)
                end
            end
        end
    end
    return coords, findall(x->x=="CYS", resnams)
end

function deform_sulfenic(coords::Vector{SVector{3,Float64}}, bits::BitVector)
    d = copy(coords)
    for idx in findall(bits)
        n = normalize(d[idx])
        d[idx] += 0.5*n
    end
    return d
end

function build_Omega_coords(states, coords, cys_idx)
    Omega = Dict{String,Vector{SVector{3,Float64}}}()
    for s in states
        bits = BitVector(c == '1' for c in collect(s))
        Omega[s] = deform_sulfenic(coords, bits)
    end
    return Omega
end


function estimate_sasa(coords, cys_idx; cutoff=5.0)
    sum(1.0 / (sum(norm(atom - coords[i]) < cutoff for atom in coords)-1 + 1e-3)
        for i in cys_idx)
end

# =============================================================================
# 4️⃣ Reactant and vibrational alignment
# =============================================================================

struct Reactant
    name::String
    coords::Vector{SVector{3,Float64}}
    interaction_radius::Float64
end

oxidant = Reactant("oxidant",
    [SVector(0.0,0.0,0.0), SVector(-1.0,0.0,0.0), SVector(0.0,-1.0,0.0)],
    5.0)

function real_deformation_energy(c1, c2)
    sqrt(sum(norm(c2[i]-c1[i])^2 for i in 1:length(c1))/length(c1))
end

function real_deformation_vector(c1, c2)
    normalize(sum(c2[i] - c1[i] for i in 1:length(c1)))
end

function reactant_orbital_vector(rxn::Reactant)
    normalize(sum(rxn.coords[i+1] - rxn.coords[i] for i in 1:length(rxn.coords)-1))
end

function cos_theta(c1, c2, rxn::Reactant)
    dot(real_deformation_vector(c1, c2), reactant_orbital_vector(rxn))
end

# =============================================================================
# 5️⃣ Saddles, total coupling, transition rate
# =============================================================================

function compute_saddle_points(G::GeoGraphReal2, f::Vector{Float64}; barrier=0.1)
    saddles = Dict{Tuple{Int,Int},Float64}()
    for i in 1:length(G.pf_states), j in G.neighbors[i]
        key = (min(i,j), max(i,j))
        saddles[key] = max(f[i], f[j]) + barrier
    end
    return saddles
end

function total_coupling_mgf(i, j, G, Omega, cys_idx, rxn, saddles)
    Rv = G.R_vals[j]
    Av = norm(G.anisotropy[j])
    ci, cj = Omega[G.pf_states[i]], Omega[G.pf_states[j]]
    def_en = real_deformation_energy(ci, cj)
    proj = def_en * cos_theta(ci, cj, rxn)
    sasaP = estimate_sasa(cj, cys_idx)
    barb = saddles[(min(i,j),max(i,j))]
    return Rv + Av + proj + sasaP + barb
end

transition_rate(i, j, G, Omega, cys_idx, rxn, saddles; α=1.0) =
    exp(-2α * total_coupling_mgf(i, j, G, Omega, cys_idx, rxn, saddles))

# =============================================================================
# 6️⃣ Final demo: run a transition from "000"
# =============================================================================

coords, cys = load_ca_and_cys("AF-P04406-F1-model_v4.pdb")
Omega = build_Omega_coords(pf_states, coords, cys)

rho = zeros(length(pf_states)); rho[1] = 1.0
update_geometry!(G, rho, coords, cys)

total_curvature = sum(G.R_vals)
println("\n🔍 Total Ricci curvature across all modes: ", round(total_curvature, digits=6))

morse_vals = [G.R_vals[i] + norm(G.anisotropy[i]) for i in 1:length(pf_states)]
saddles = compute_saddle_points(G, morse_vals)

println("Transition rates from '000':")
for j in G.neighbors[1]
    r = transition_rate(1, j, G, Omega, cys, oxidant, saddles; α=1.0)
    println("  000 → $(pf_states[j]): rate = ", round(r, sigdigits=6))
end

println("\nRicci curvature R(x) and anisotropy A(x):")
for i in 1:length(pf_states)
    R_val = round(G.R_vals[i], digits=4)
    A_val = round(norm(G.anisotropy[i]), digits=4)
    println("  State $(pf_states[i]): R = $R_val, A = $A_val")
end

println("\n3D Coordinates of Cysteines:")
for i in cys
    println("  CYS index $i: ", coords[i])
end
