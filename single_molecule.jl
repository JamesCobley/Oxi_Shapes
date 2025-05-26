# =============================================================================
# Redox Geometry Model (Standalone, Cohesive)
# =============================================================================

using StaticArrays
using LinearAlgebra

# -- Step 1: Load real coordinates from an AlphaFold PDB file --
function load_ca_trace_and_cysteines(pdb_path::String)
    coords = SVector{3, Float64}[]
    resnames = String[]

    open(pdb_path, "r") do io
        for line in eachline(io)
            startswith(line, "ATOM") || continue
            atom_name = strip(line[13:16])
            resname   = strip(line[18:20])
            if atom_name == "CA"
                x = parse(Float64, line[31:38])
                y = parse(Float64, line[39:46])
                z = parse(Float64, line[47:54])
                push!(coords, SVector(x, y, z))
                push!(resnames, resname)
            end
        end
    end

    cys_indices = findall(x -> x == "CYS", resnames)
    return coords, cys_indices
end

# -- Step 2: Define the redox-deformed shape generator Ω(x) --
function generate_Omega_x(coords::Vector{SVector{3, Float64}},
                          cys_indices::Vector{Int},
                          istate::BitVector)
    coords_new = copy(coords)
    for (i, idx) in enumerate(cys_indices)
        perturb = istate[i] == 0 ?
                  0.8 .* (rand(SVector{3, Float64}) .- 0.5) :  # reduced
                  0.2 .* (rand(SVector{3, Float64}) .- 0.5)    # oxidized
        coords_new[idx] += perturb
    end
    return coords_new
end

# -- Step 3: RMSD function for comparing shapes --
function rmsd(c1::Vector{SVector{3, Float64}}, c2::Vector{SVector{3, Float64}})
    sqrt(sum(norm(c1[i] - c2[i])^2 for i in 1:length(c1)) / length(c1))
end

# -- Step 4: Define standalone redox geometry object --
mutable struct OmegaReal1
    coords::Vector{SVector{3, Float64}}         # original real shape
    cys_indices::Vector{Int}                    # positions of CYS
    redox_state::BitVector                      # redox i-state
    deformed::Vector{SVector{3, Float64}}       # Ω(x)
    deformation_energy::Float64                 # ||Ω(x) - coords||
end

function OmegaReal1(pdb_path::String, istate::BitVector)
    coords, cys_indices = load_ca_trace_and_cysteines(pdb_path)
    deformed = generate_Omega_x(coords, cys_indices, istate)
    energy = rmsd(coords, deformed)
    return OmegaReal1(coords, cys_indices, istate, deformed, energy)
end

# =============================================================================
# Example use
# =============================================================================

pdb_path = "/content/AF-P04406-F1-model_v4.pdb"
istate = BitVector([1, 0, 1])  # Example redox state: oxidized, reduced, oxidized

Ω = OmegaReal1(pdb_path, istate)

println("Redox state: ", Ω.redox_state)
println("Deformation energy: ", Ω.deformation_energy)
println("Cysteine positions: ", Ω.cys_indices)
println("Original Cys1 coord: ", Ω.coords[Ω.cys_indices[1]])
println("Deformed Cys1 coord: ", Ω.deformed[Ω.cys_indices[1]])
