export solve, QZ, QuadraticIteration

function jacobian_sequence(
    model::RationalExpectationsModel, ss, ε; backend=AutoForwardDiff()
)
    return cat(
        DifferentiationInterface.jacobian(x -> model([ss ss x], ε), backend, ss),
        DifferentiationInterface.jacobian(x -> model([ss x ss], ε), backend, ss),
        DifferentiationInterface.jacobian(x -> model([x ss ss], ε), backend, ss),
        dims = 3
    )
end

function solve(model::RationalExpectationsModel, order::Int=1; kwargs...)
    return solve(model, Val(order); kwargs...)
end

function solve(model::RationalExpectationsModel, order; kwargs...)
    return error("only first order perturbation methods are defined")
end

include("perturbation_methods/first_order.jl")

# only the linear Gaussian state space model is defined here
function SSMProblems.StateSpaceModel(
    model::RationalExpectationsModel, obs_idx, ση²; kwargs...
)
    A, B = solve(p, 1; kwargs...)
    (nx, nε), ny = size(B), length(obs_idx)
    @assert ny ≤ nε "more observables than shocks"

    C = diagm(ones(Bool, nx))[obs_idx, :]
    return linear_gaussian_control(A, B, C, ση² * I(ny))
end
