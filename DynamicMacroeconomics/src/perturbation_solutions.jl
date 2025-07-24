export solve, QuadraticIteration, FirstOrderPerturbation

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

## FIRST ORDER #############################################################################

"""
    FirstOrderPerturbation

A first order approximation of a given rational expectations model which stores the state
jacobians `∂Y` and the shock jacobian `∂E`.
"""
struct FirstOrderPerturbation
    ∂Y::AbstractArray{<:Real,3}
    ∂E::AbstractArray{<:Real,2}
    function FirstOrderPerturbation(
        model::SteadyStateModel{N}, state_variables, shock_variables
    ) where {N}
        # collect jacobians as a 3 dimensional array
        base_model, steady_state = model.base_model, model.steady_state
        partials = merge([get_partials(base_model[i], steady_state) for i in 1:N]...)
        system = []
        for target in base_model.targets
            ∂y = get.(Ref(partials[target]), state_variables, Ref(zeros(Bool, 3)))
            push!(system, cat(∂y..., dims=2))
        end

        # construct shocks from targets
        shocks = get_shocks(base_model, shock_variables)
        return new(permutedims(cat(system..., dims=3), (3, 2, 1)), shocks)
    end
end

# TODO: confirm this isn't total dogshit (I think it may be incorrect)
function get_shocks(model::GraphicalModel, shock_variables)
    shocks = zeros(length(model.targets), 1)
    shocks[indexin(shock_variables, collect(model.targets))] .= -1
    return shocks
end

"""
    QuadraticIteration(; tol=1e-12, max_iters=2^10)

A brute force algorithm which uses an iterative root finding scheme to converge to a policy
function. While this approach is much simpler than the QZ decomposition, it is considerably
slower and less accurate. Despite its flaws, it is fully differentiable via AD, and is the
correct choice for running HMC estimation.

# Example

Given a rational expectations model `model`, we solve the first order system like so:

```julia
system = FirstOrderPerturbation(model);
policy, impact = solve(system, QuadraticIteration());
```

See also [`QZ`](@ref).
"""
Base.@kwdef struct QuadraticIteration
    tol::Real = 1e-12
    max_iters::Int = 2^10
end

function solve(system::FirstOrderPerturbation, algo::QuadraticIteration)
    C, B, A = eachslice(system.∂Y, dims=3)
    ghx = zero(A)

    for _ in 1:algo.max_iters
        ghx = -(A * ghx + B) \ C
        if maximum(C + B * ghx + A * ghx * ghx) < algo.tol
            break
        end
    end

    return ghx, (A * ghx + B) \ -system.∂E
end

function solve(
    model::SteadyStateModel, states, shocks, ::Val{1}; algo=QuadraticIteration(), kwargs...
)
    system = FirstOrderPerturbation(model, states, shocks)
    return solve(system, algo)
end
