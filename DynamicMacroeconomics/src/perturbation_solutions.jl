export solve, QZ, QuadraticIteration

using DifferentiationInterface: jacobian

function jacobian_sequence(
    model::RationalExpectationsModel, parameters; backend=AutoForwardDiff()
)
    nil_shocks = zeros(length(model.shocks), 3)
    ss = steady_state(model, parameters)
    return cat(
        jacobian(x -> model([ss ss x], nil_shocks, parameters), backend, ss),
        jacobian(x -> model([ss x ss], nil_shocks, parameters), backend, ss),
        jacobian(x -> model([x ss ss], nil_shocks, parameters), backend, ss),
        dims = 3
    )
end

"""
    solve(model, parameters, order; kwargs...)

For a given rational expectations model, solve the k-th order approximation to obtain the
policy function for Markov representation.

See also [`state_space`](@ref)..
"""
function solve(model::RationalExpectationsModel, parameters, order::Int=1; kwargs...)
    return solve(model, parameters, Val(order); kwargs...)
end

function solve(model::RationalExpectationsModel, parameters, order; kwargs...)
    return error("only first order perturbation methods are defined")
end

include("perturbation_methods/first_order.jl")
