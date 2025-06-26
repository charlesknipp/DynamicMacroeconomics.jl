using LinearAlgebra, MatrixEquations, OffsetArrays
using SSMProblems, GeneralisedFilters
using Distributions
using Turing
using StatsPlots

## DFM EXAMPLE #############################################################################

function dynamic_factor_model(loadings::Vector{T}) where {T<:Real}
    A = [0.85;;]
    Q = Diagonal([0.4])
    H = reshape([1; loadings], length(loadings)+1, 1)
    R = Diagonal([0.3, 0.4, 0.5, 0.4, 0.6])

    # recall that lyapd is non-differentiable (copy rrule from ControlSystems.jl)
    return create_homogeneous_linear_gaussian_model(
        zeros(T, 1), lyapd(A, Q), A, zeros(T, size(A, 1)), Q, H, zeros(T, size(H, 1)), R
    )
end

true_model = dynamic_factor_model([0.8, -0.3, 0.6, 1.2]);
_, ys = sample(true_model, 250);

## DIRECT ITERATION SOLVER #################################################################

@model function dfm_direct_iteration(data)
    λs ~ MvNormal(0.1I(4))
    ssm = dynamic_factor_model(λs)

    # sample from the state space model directly
    x0 ~ SSMProblems.distribution(ssm.prior)
    x = OffsetVector(fill(x0, length(data) + 1), -1)
    for t in eachindex(data)
        x[t] ~ SSMProblems.distribution(ssm.dyn, t, x[t-1])
        data[t] ~ SSMProblems.distribution(ssm.obs, t, x[t])
    end
end

# slower, but more accurate direct iteration solver
chain_1 = sample(dfm_direct_iteration(ys), NUTS(), MCMCThreads(), 500, 5)
plot(group(chain_1, :λs))

## MARGINAL LIKELIHOOD SOLVER ##############################################################

@model function dfm_marginalization(data)
    λs ~ MvNormal(0.1I(4))
    ssm = dynamic_factor_model(λs)

    # run the Kalman filter (can be any filter)
    _, logZ = GeneralisedFilters.filter(ssm, KF(), data)
    Turing.@addlogprob! logZ
end

# this is much faster for linear gaussian state space models
chain_2 = sample(dfm_marginalization(ys), NUTS(), MCMCThreads(), 500, 5)
plot(group(chain_2, :λs))
