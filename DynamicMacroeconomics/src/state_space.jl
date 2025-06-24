## LINEAR GAUSSIAN MODEL ###################################################################

struct LinearGaussianControllableDynamics{AT,BT} <: LinearGaussianLatentDynamics
    A::AT
    B::BT
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

struct LinearGaussianControllableObservation{CT,DT} <: LinearGaussianObservationProcess
    C::CT
    D::DT
end

function GeneralisedFilters.calc_H(
    obs::LinearGaussianControllableObservation, ::Int; kwargs...
)
    return obs.C
end

function GeneralisedFilters.calc_c(
    obs::LinearGaussianControllableObservation, ::Int; kwargs...
)
    return zeros(Bool, size(obs.C, 1))
end

function GeneralisedFilters.calc_R(
    obs::LinearGaussianControllableObservation, ::Int; kwargs...
)
    return obs.D * obs.D'
end

function SSMProblems.distribution(
    obs::LinearGaussianControllableObservation, step::Integer, state::AbstractVector; kwargs...
)
    H = GeneralisedFilters.calc_H(obs, step; kwargs...)
    return StructuralMvNormal(H * state, obs.D)
end

function linear_gaussian_control(A, B, C, D)
    T = Base.promote_eltype(A, B)
    # Σ = lyapd(A, B * B' + 1e-12I)
    Σ = I(size(A, 1))
    return SSMProblems.StateSpaceModel(
        GeneralisedFilters.HomogeneousGaussianPrior(zeros(T, size(A, 1)), Σ),
        LinearGaussianControllableDynamics(A, B),
        LinearGaussianControllableObservation(C, D),
    )
end
