using Flux
using CUDA
using LinearAlgebra
using Random
using Statistics
using Plots

# Try using GPU if available, fallback to CPU
const DEVICE = CUDA.has_cuda() ? CUDADevice() : CPU()
const FLOATX = CUDA.has_cuda() ? CuArray{Float32} : Array{Float32}

if DEVICE isa CUDADevice
    println("🚀 Device: GPU")
else
    println("🧠 Device: CPU (CUDA not available)")
end

# Differentiable λ parameter (stored in log-space for stability)
lambda_net = Flux.param([log(1.0f0)])

# Getter function for the actual λ value
function get_lambda()
    return exp(lambda_net[1])
end

# Global RT constant (same as in PyTorch)
const RT = 1.0f0

using Graphs
using Delaunay
using LinearAlgebra
using Plots  # or Makie if you prefer GPU plots

# --- Define states and connectivity
pf_states = ["000", "001", "010", "100", "011", "101", "110", "111"]
allowed_edges = [
    ("000", "001"), ("000", "010"), ("000", "100"),
    ("001", "101"), ("001", "011"),
    ("010", "110"), ("010", "011"),
    ("011", "111"),
    ("100", "110"), ("100", "101"),
    ("101", "111"),
    ("110", "111")
]

num_states = length(pf_states)
state_index = Dict(s => i for (i, s) in enumerate(pf_states))
index_state = Dict(i => s for (s, i) in state_index)

# --- Create graph
G = SimpleGraph(num_states)
for (u, v) in allowed_edges
    add_edge!(G, state_index[u], state_index[v])
end

# --- Flat diamond layout (2D positions)
flat_pos = Dict(
    "000" => [0.0, 3.0],
    "001" => [-2.0, 2.0],
    "010" => [0.0, 2.0],
    "100" => [2.0, 2.0],
    "011" => [-1.0, 1.0],
    "101" => [0.0, 1.0],
    "110" => [1.0, 1.0],
    "111" => [0.0, 0.0]
)

# --- Node positions as matrix (each row: [x, y])
node_xy = reduce(vcat, [reshape(flat_pos[s], 1, 2) for s in pf_states])
triangulation = delaunay(node_xy')  # transpose for correct orientation
triangles = triangulation.triangles  # matrix with column = triangle (3 indices)

# --- Precompute neighbor indices (index-based)
neighbor_indices = Dict{String, Vector{Int}}()
for s in pf_states
    neighbors = []
    for (u, v) in allowed_edges
        if u == s
            push!(neighbors, state_index[v])
        elseif v == s
            push!(neighbors, state_index[u])
        end
    end
    neighbor_indices[s] = neighbors
end

# --- Optional GPU-ready flat position tensor
flat_pos_tensor = Float32.(node_xy)  # [8 x 2] matrix
# If you're using CUDA, do: `flat_pos_tensor = cu(flat_pos_tensor)`

# --- Visualization (Plots.jl example)
scatter(node_xy[:,1], node_xy[:,2], label="States", color=:skyblue, markersize=8)
for (u, v) in allowed_edges
    i, j = state_index[u], state_index[v]
    plot!([node_xy[i,1], node_xy[j,1]], [node_xy[i,2], node_xy[j,2]], color=:gray, lw=2)
end
title!("Proteoform State Space (R=3)")
xlabel!("X")
ylabel!("Y")
display(current())  # show plot
