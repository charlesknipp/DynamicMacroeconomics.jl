export solve, QuadraticIteration, SequenceJacobian

using DifferentiationInterface: jacobian

"""
    solve(model, parameters, order; kwargs...)

For a given rational expectations model, solve the k-th order approximation to obtain the
policy function for Markov representation.

See also [`state_space`](@ref)..
"""
function solve(model::SteadyStateModel, states, shocks, order::Int=1; kwargs...)
    return solve(model, states, shocks, Val(order); kwargs...)
end

function solve(model::SteadyStateModel, states, shocks, order; kwargs...)
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
system = FirstOrderSystem(model, statevars, shockvars);
policy, impact = solve(system, QuadraticIteration());
```

See also [`QZ`](@ref).
"""
Base.@kwdef struct QuadraticIteration
    tol::Real = 1e-12
    max_iters::Int = 2^10
end

function solve(system::FirstOrderSystem, algo::QuadraticIteration)
    C, B, A = eachslice(system.∂Z, dims=3)
    ghx = zero(A)

    for _ in 1:algo.max_iters
        ghx = -(A * ghx + B) \ C
        if maximum(C + B * ghx + A * ghx * ghx) < algo.tol
            break
        end
    end

    return ghx, (A * ghx + B) \ -system.∂U
end

function solve(
    model::SteadyStateModel, states, shocks, ::Val{1}; algo=QuadraticIteration(), kwargs...
)
    system = FirstOrderSystem(model, states, shocks)
    return solve(system, algo)
end

## SEQUENCE SPACE METHODS ##################################################################

function convmat(basis::AbstractVector{T}, N) where {T<:Real}
    padding = zeros(T, N ≥ 3 ? N - 2 : 0)
    return Toeplitz([basis[2:-1:1]; padding], [basis[2:end]; padding])
end

function convmat(basis::AbstractArray{T, 3}, N) where {T<:Real}
    nx, ny, _ = size(basis)
    M = zeros(T, nx * N, ny * N)
    for i in 1:nx, j in 1:ny
        M[(N*(i-1)+1):N*i, N*(j-1)+1:N*j] = convmat(basis[i, j, :], N)
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
system = FirstOrderSystem(model, statevars, shockvars);
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
    HU = convmat(system.∂Z, algo.T)
    HZ = convmat(system.∂U, algo.T)

    # TODO: the reshaping is so beyond untested, but eh it's okay
    G = reshape(-HU \ HZ, (algo.T, size(system.∂Z, 2), algo.T))
    return permutedims(G, (2, 1, 3))
end
