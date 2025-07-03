using LinearAlgebra, MatrixEquations, OffsetArrays
using SSMProblems, GeneralisedFilters
using Distributions
using Turing

# for plotting MCMC chains with Makie.jl instead of Plots.jl
include("../utilities/mcmc_plots.jl");

## DFM EXAMPLE #############################################################################

# in the original code Babur originally set R = Diagonal([0.3, 0.4, 0.5, 0.4, 0.6]) and
# λs = [0.8, -0.3, 0.6, 1.2], where Λ = [1; λs]

function dynamic_factor_model(loadings::Vector{ΛT}, σ::ΣT) where {ΛT<:Real, ΣT<:Real}
    ny = length(loadings) + 1
    nx = 1

    # transition process is mean reverting random walk
    A = [0.85;;]
    Q = Diagonal([0.4])

    # factor loading normalized such that Λ[1] = 1 and measurement noise is iid
    Λ = reshape([1; loadings], ny, 1)
    Σ = Diagonal(σ * ones(ny))

    # recall that lyapd is non-differentiable (copy rrule from ControlSystems.jl)
    return create_homogeneous_linear_gaussian_model(
        zeros(ΛT, 1), lyapd(A, Q), A, zeros(ΛT, nx), Q, Λ, zeros(ΣT, ny), Σ
    )
end

## DIRECT ITERATION SOLVER #################################################################

@model function dfm_direct_iteration(data)
    λs ~ MvNormal(0.1I(length(data[1]) - 1))
    ssm = dynamic_factor_model(λs, 0.2)

    # sample from the state space model directly
    x0 ~ SSMProblems.distribution(ssm.prior)
    x = OffsetVector(fill(x0, length(data) + 1), -1)
    for t in eachindex(data)
        x[t] ~ SSMProblems.distribution(ssm.dyn, t, x[t-1])
        data[t] ~ SSMProblems.distribution(ssm.obs, t, x[t])
    end
end

## MARGINAL LIKELIHOOD SOLVER ##############################################################

@model function dfm_marginalization(data)
    λs ~ MvNormal(0.1I(length(data[1]) - 1))
    ssm = dynamic_factor_model(λs, 0.2)

    # run a filtering algorithm (here we use the Kalman filter)
    _, logZ = GeneralisedFilters.filter(ssm, KF(), data)
    Turing.@addlogprob! logZ
end

## BENCHMARKS ##############################################################################

# data generating process with low measurement noise
true_λs = randn(9);
true_model = dynamic_factor_model(true_λs, 0.2);
_, ys = sample(true_model, 250);

# 1813.56 seconds
chain_1 = sample(dfm_direct_iteration(ys), NUTS(), MCMCThreads(), 500, 3);
plot(group(chain_1, :λs), true_λs; size=(600, 1200))

# 92.96 seconds
chain_2 = sample(dfm_marginalization(ys), NUTS(), MCMCThreads(), 500, 3);
plot(group(chain_2, :λs), true_λs; size=(600, 1200))
