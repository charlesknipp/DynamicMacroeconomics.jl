export state_space, LinearGaussianControllableDynamics

struct LinearGaussianControllableDynamics <: LinearGaussianLatentDynamics
    A
    B
end

function GeneralisedFilters.calc_A(
    dyn::LinearGaussianControllableDynamics, ::Int; kwargs...
)
    return dyn.A
end

function GeneralisedFilters.calc_b(
    dyn::LinearGaussianControllableDynamics, ::Int; kwargs...
)
    return zeros(Bool, size(dyn.A, 1))
end

function GeneralisedFilters.calc_Q(
    dyn::LinearGaussianControllableDynamics, ::Int; kwargs...
)
    return dyn.B * dyn.B'
end

function SSMProblems.distribution(
    dyn::LinearGaussianControllableDynamics, step::Integer, state::AbstractVector; kwargs...
)
    A = GeneralisedFilters.calc_A(dyn, step; kwargs...)
    return StructuralMvNormal(A * state, dyn.B)
end

"""
    state_space(model, parameters, observations, order; noise, kwargs...)

Construct a state space model from a given rational expectations model `model` observing
variables set within `observations` with noise `noise`. Approximation of the policy function
is set by default to order = 1.

See also [`solve`](@ref)..
"""
function state_space(
    model::GraphicalModel,
    parameters,
    observations,
    order::Int = 1;
    noise = 1.0,
    kwargs...
)
    return state_space(model, parameters, observations, Val(order), noise; kwargs...)
end

function state_space(
    model::GraphicalModel, parameters, observations, order, noise; kwargs...
)
    error("higher order state space models not yet supported")
end

# # TODO: define a rrule for `lyapd` to use analytical covariance
# function state_space(
#     model::GraphicalModel, parameters, observations, ::Val{1}, noise; kwargs...
# )
#     nx, ny = length(model.states), length(observations)
#     A, B = solve(model, parameters, 1; kwargs...)
#     C = I(nx)[indexin(observations, [model.states...]), :]
#     T = Base.promote_eltype(A, B)
#     # Σ0 = lyapd(A, B * B' + 1e-12I)

#     return SSMProblems.StateSpaceModel(
#         GeneralisedFilters.HomogeneousGaussianPrior(zeros(T, nx), I(nx)),
#         LinearGaussianControllableDynamics(A, B),
#         GeneralisedFilters.HomogeneousLinearGaussianObservationProcess(
#             C, zeros(Bool, ny), noise * I(ny)
#         )
#     )
# end
