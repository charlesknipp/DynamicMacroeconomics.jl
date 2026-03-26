export solve, QuadraticIteration, SequenceJacobian, QZ

using DifferentiationInterface: jacobian

"""
    solve(model, steady_state, states, shocks, order; kwargs...)

For a given rational expectations model, solve the k-th order approximation to obtain the
policy function for Markov representation.

See also [`state_space`](@ref)..
"""
function solve(block::AbstractBlock, ss, states, shocks, order::Int=1; kwargs...)
    return solve(block, ss, states, shocks, Val(order); kwargs...)
end

function solve(block::AbstractBlock, ss, states, shocks, order; kwargs...)
    return error("only first order perturbation methods are defined")
end

## PERTURBATION METHODS ####################################################################

"""
    QuadraticIteration(; tol=1e-12, max_iters=2^10)

A brute force algorithm which uses an iterative root finding scheme to converge to a policy
function. While this approach is much simpler than the QZ decomposition, it is considerably
slower and less accurate. Despite its flaws, it is fully differentiable via AD, and is the
correct choice for running HMC estimation.

# Example

Given a rational expectations model `model`, we solve the first order system like so:

```julia
system = FirstOrderSystem(model, steady_state, states, shocks, targets);
policy, impact = solve(system, QuadraticIteration());
```

See also [`QZ`](@ref).
"""
Base.@kwdef struct QuadraticIteration
    tol::Real = 1e-12
    max_iters::Int = 2^10
end

function solve(system::FirstOrderSystem, algo::QuadraticIteration)
    C, B, A = eachslice(system.∂U; dims=3)
    ghx = zero(A)

    for _ in 1:(algo.max_iters)
        ghx = -(A * ghx + B) \ C
        if maximum(C + B * ghx + A * ghx * ghx) < algo.tol
            break
        end
    end

    return ghx, (A * ghx + B) \ -system.∂Z[:, :, 2]
end

function solve(
    model::AbstractBlock, ss, states, shocks, ::Val{1}; algo=QuadraticIteration(), kwargs...
)
    system = FirstOrderSystem(model, ss, states, shocks)
    return solve(system, algo)
end

"""
    QZ()

An algorithm reliant on the ordered Schur decomposition of system matrices in recursive form
which is fully generalized to systems containing one lead and one lag. This is considerably
faster than linear time iteration, but induces problems with automatic differentiation.

This algorithm takes a first order system of the following form:
```math
A x_{t+1} + B x_{t} + C x_{t-1} + D u_{t} = 0
```
to solve for the policy function ``P`` such that
```math
x_{t} = P x_{t+1} + Q u_{t}
```

This particular implementation, based on (Auclert et al, 2025), neatly transforms our system
which solves a matrix quadratic using QZ.

```math
\begin{bmatrix} I & 0 \\ 0 & A \end{bmatrix} \begin{bmatrix} I \\ P \end{bmatrix} P =
\begin{bmatrix} 0 & I \\ -C & -B \end{bmatrix} \begin{bmatrix} I \\ P \end{bmatrix}
```

# Example

Given a rational expectations model `model`, we solve the first order system like so:

```julia
system = FirstOrderSystem(model, steady_state, states, shocks, targets);
policy, impact = solve(system, QZ());
```

See also [`QuadraticIteration`](@ref).
"""
struct QZ end

function solve(system::FirstOrderSystem, ::QZ)
    C, B, A = eachslice(system.∂U, dims=3)
    n = size(system.∂U, 1)

    F = schur([zero(A) I(n); -C -B], cat(I(n), A, dims=(1, 2)))
    eigenvalues = F.α ./ F.β

    stable_flag = abs.(eigenvalues) .< 1
    nstable = count(stable_flag)
    ordschur!(F, stable_flag)

    Z11 = F.Z[1:nstable, 1:nstable]
    Z21 = F.Z[nstable+1:end, 1:nstable]

    if rank(Z11) < nstable
        warn("Invertibility condition violated")
    end

    P = Z21 * inv(Z11)
    return P, (A * P + B) \ -system.∂Z[:, :, 2]
end

## SEQUENCE SPACE METHODS ##################################################################

function convmat(basis::AbstractVector{T}, N) where {T<:Real}
    padding = zeros(T, N ≥ 3 ? N - 2 : 0)
    return Toeplitz([basis[2:-1:1]; padding], [basis[2:end]; padding])
end

function convmat(basis::AbstractArray{T,3}, N) where {T<:Real}
    nx, ny, _ = size(basis)
    M = zeros(T, nx * N, ny * N)
    for i in 1:nx, j in 1:ny
        M[(N * (i - 1) + 1):(N * i), (N * (j - 1) + 1):(N * j)] = convmat(basis[i, j, :], N)
    end
    return M
end

"""
    SequenceJacobian

Solve a given model in the sequence space by expanding it's first order system into a MA(T)
which should be more efficient for heterogeneous models.

# Example

Given a rational expectations model `model`, we solve the first order system in the sequence
space like so:

```julia
system = FirstOrderSystem(model, steady_state, states, shocks, targets);
sequence_jacobians = solve(system, SequenceJacobian(150));
```

The output `sequence_jacobians` is a num_states × T × T array, which represents the model's
impulse responses to news shocks at time t ∈ 1:T.

See also [`QuadraticIteration`](@ref).
"""
Base.@kwdef struct SequenceJacobian
    T::Int = 300
end

function solve(system::FirstOrderSystem, algo::SequenceJacobian)
    HU = convmat(system.∂U, algo.T)
    HZ = convmat(system.∂Z, algo.T)

    G = reshape(-HU \ HZ, (algo.T, size(system.∂U, 2), algo.T))
    return permutedims(G, (2, 1, 3))
end
