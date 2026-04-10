export solve, QuadraticIteration, SequenceJacobian, QZ

using DifferentiationInterface: jacobian

"""
    solve(model, steady_state, unknowns, shocks, targets; order=1, kwargs...)

For a given rational expectations model, solve the k-th order approximation to obtain the
general equilibrium solution either in state space or sequence space depending on the given
algorithm.

See also [`state_space`](@ref)..
"""
function solve(
    block::AbstractBlock,
    ss,
    unknowns,
    shocks,
    targets=outputs(block);
    order::Int=1,
    kwargs...
)
    return solve(block, ss, unknowns, shocks, targets, Val(order); kwargs...)
end

# solving to higher orders
function solve(block::AbstractBlock, ss, unknowns, shocks, targets, order; kwargs...)
    return error("only first order perturbation methods are defined")
end

# solving to a first order
function solve(
    model::AbstractBlock,
    ss,
    unknowns,
    shocks,
    targets,
    ::Val{1};
    algo=QuadraticIteration(),
    kwargs...
)
    J = jacobian(model, ss, union(unknowns, shocks), targets; kwargs...)
    return solve(J, shocks, algo)
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
block_jacobian = jacobian(model, steady_state, states, targets);
policy, impact = solve(block_jacobian, shocks, QuadraticIteration());
```

See also [`QZ`](@ref).
"""
Base.@kwdef struct QuadraticIteration
    tol::Real = 1e-12
    max_iters::Int = 2^10
end

function solve(A::BlockJacobian, controls, algo::QuadraticIteration)
    states = symdiff(inputs(A), controls)
    ∂U = subset(A, :, states)
    ∂Z = subset(A, :, controls)

    ∂U1 = firstband(∂U)
    P = zero(∂U1)
    system = matrix_polynomial(∂U, P)

    for _ in 1:(algo.max_iters)
        P = -system \ ∂U1
        system = matrix_polynomial(∂U, P)
        if maximum(∂U1 + system * P) < algo.tol
            break
        end
    end

    return P, system \ -getband(∂Z, 0)
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

This particular implementation, based on (Anderson, 2006), neatly transforms our system
which solves a matrix quadratic using QZ.

```math
\begin{bmatrix} I & 0 \\ 0 & A \end{bmatrix} \begin{bmatrix} I \\ P \end{bmatrix} P =
\begin{bmatrix} 0 & I \\ -C & -B \end{bmatrix} \begin{bmatrix} I \\ P \end{bmatrix}
```

# Example

Given a rational expectations model `model`, we solve the first order system like so:

```julia
block_jacobian = jacobian(model, steady_state, states, targets);
policy, impact = solve(block_jacobian, shocks, QZ());
```

See also [`QuadraticIteration`](@ref).
"""
struct QZ end

# not sure if this is exactly the interface I want, but it works for now
function LinearAlgebra.schur(A::BlockJacobian)
    As..., A1 = eachoffset(A)
    n = size(A1, 1)
    m = length(As) - 1
    return schur(
        [zeros(n * m, n) I(n * m); -hcat(As...)], cat(I(n * m), A1, dims=(1, 2))
    )
end

# TODO: there is likely a better approach
function matrix_polynomial(A::BlockJacobian, P::AbstractMatrix)
    idx = offset_range(A)[2:end]
    return sum(ntuple(i -> getband(A, idx[i]) * (P ^ (i - 1)), length(idx)))
end

function solve(A::BlockJacobian, controls, ::QZ)
    states = symdiff(inputs(A), controls)
    ∂U = subset(A, :, states)
    ∂Z = subset(A, :, controls)

    F = schur(∂U)
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
    return P, matrix_polynomial(∂U, P) \ -getband(∂Z, 0)
end

## SEQUENCE SPACE METHODS ##################################################################

"""
    SequenceJacobian

Solve a given model in the sequence space by expanding it's first order system into a MA(T)
which should be more efficient for heterogeneous models.

# Example

Given a rational expectations model `model`, we solve the first order system in the sequence
space like so:

```julia
block_jacobian = jacobian(model, steady_state, states, targets);
sequence_jacobians = solve(block_jacobian, shocks, SequenceJacobian(150));
```

The output `sequence_jacobians` is a num_states × T × T array, which represents the model's
impulse responses to news shocks at time t ∈ 1:T.

See also [`QuadraticIteration`](@ref).
"""
Base.@kwdef struct SequenceJacobian
    T::Int = 300
end

function sequence_jacobian(A::BlockJacobian, T::Integer)
    # this is rather disgusting, but it works for the time being
    return vcat(hcat.(eachcol(Toeplitz.(A.partials, T))...)...)
end

function solve(A::BlockJacobian, controls, algo::SequenceJacobian)
    states = symdiff(inputs(A), controls)
    ∂U = subset(A, :, states)
    ∂Z = subset(A, :, controls)
    return -sequence_jacobian(∂U, algo.T) \ sequence_jacobian(∂Z, algo.T)
end
