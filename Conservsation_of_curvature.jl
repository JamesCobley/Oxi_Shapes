#!/usr/bin/env julia
# ===============================================================
# Oxi-Shapes (R=3 example, e.g., GAPDH): Zero-Sum Curvature Test
# ===============================================================

using LinearAlgebra
using Random

# ---------------------------
# Boolean lattice utilities
# ---------------------------

"All R-bit states as zero-padded binary strings."
function bitstates(R::Int)
    return [lpad(string(i, base=2), R, '0') for i in 0:(2^R - 1)]
end

"Hamming distance between two equal-length bitstrings."
hamming(a::AbstractString, b::AbstractString) = sum(c1 != c2 for (c1, c2) in zip(a, b))

"Edges between states that differ by exactly 1 bit (first-order adjacency)."
function hamming1_edges(states::Vector{String})
    E = Vector{Tuple{Int,Int}}()
    n = length(states)
    for i in 1:n-1, j in i+1:n
        if hamming(states[i], states[j]) == 1
            push!(E, (i, j))  # undirected edge
        end
    end
    return E
end

"Symmetric adjacency matrix from undirected edges (unit weights)."
function adjacency(n::Int, edges::Vector{Tuple{Int,Int}})
    W = zeros(Float64, n, n)
    for (i, j) in edges
        W[i, j] = 1.0
        W[j, i] = 1.0
    end
    return W
end

"Combinatorial graph Laplacian L = D - W."
laplacian(W::AbstractMatrix) = Diagonal(vec(sum(W, dims=2))) - W

"Curvature field R = Δρ (optionally scaled by λ)."
c_ricci(L::AbstractMatrix, ρ::AbstractVector; λ::Float64=1.0) = λ .* (L * ρ)

# ---------------------------
# Pretty printing
# ---------------------------

function print_labeled_vector(label::AbstractString, states, v::AbstractVector; digits=6)
    println(label)
    for i in 1:length(states)
        val = round(v[i], digits=digits)
        println("  ", states[i], " : ", val)
    end
end

function print_neighbors(states, W)
    n = length(states)
    println("Degrees and neighbors:")
    for i in 1:n
        nbr_idx = findall(j -> W[i,j] ≈ 1.0, 1:n)
        nbrs = join(states[nbr_idx], ", ")
        println("  ", states[i], "  deg=", length(nbr_idx), "  ->  [", nbrs, "]")
    end
end

# ---------------------------
# Main test
# ---------------------------

function main(; R::Int=3, λ::Float64=1.0, tol::Float64=1e-12)
    println("=== Oxi-Shapes Curvature Zero-Sum Test ===")
    println("R = ", R, "  (#states = ", 2^R, ")\n")

    states = bitstates(R)
    edges  = hamming1_edges(states)
    n      = length(states)
    W      = adjacency(n, edges)
    L      = laplacian(W)

    print_neighbors(states, W)
    println()

    println("Single-molecule placements (unit mass at exactly one mode):")
    for i in 1:n
        ρ = zeros(Float64, n)
        ρ[i] = 1.0
        Rfield = c_ricci(L, ρ; λ=λ)
        S = sum(Rfield)

        println("\n-- Placement at state ", states[i], " --")
        print_labeled_vector("Curvature R = Δρ:", states, Rfield; digits=6)
        println("Sum of curvature Σ R = ", round(S, digits=12),
                "   ==> ", abs(S) ≤ tol ? "PASS (≈ 0)" : "FAIL")
    end

    println("\nRandomized sanity check (5 draws):")
    for t in 1:5
        ρ = rand(n); ρ ./= sum(ρ)
        Rfield = c_ricci(L, ρ; λ=λ)
        S = sum(Rfield)
        println("  Draw ", t, ": Σ R = ", round(S, digits=12),
                "   ", abs(S) ≤ tol ? "PASS" : "FAIL")
    end

    println("\nConclusion: Σ_x Δρ(x) = 0 always, by Laplacian symmetry + volume invariance.")
end

# Run
main(R=3, λ=1.0)
