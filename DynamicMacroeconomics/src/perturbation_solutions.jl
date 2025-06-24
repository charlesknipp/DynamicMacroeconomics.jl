export solve

function solve(p::RationalExpectationsModel, order::Int=1; kwargs...)
    return solve(p, Val(order); kwargs...)
end

function solve(p::RationalExpectationsModel, order; kwargs...)
    return error("only first order perturbation methods are defined")
end

include("perturbation_methods/first_order.jl")

# only the linear Gaussian state space model is defined here
function SSMProblems.StateSpaceModel(
    p::RationalExpectationsModel, obs_idx, ση²; kwargs...
)
    A, B = solve(p, 1; kwargs...)
    (nx, nε), ny = size(B), length(obs_idx)
    @assert ny ≤ nε "more observables than shocks"

    C = diagm(ones(Bool, nx))[obs_idx, :]
    return linear_gaussian_control(A, B, C, ση² * I(ny))
end
