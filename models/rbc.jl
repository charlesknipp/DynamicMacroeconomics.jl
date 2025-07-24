using DynamicMacroeconomics
using Turing, Distributions, Random
using GeneralisedFilters

@simple function households(C, K, Z, α, β, γ, δ)
    euler = (C ^ -γ) - (β * (α * lead(Z) * K ^ (α - 1) + 1 - δ) * (lead(C) ^ -γ))
    return euler
end

@simple function firms(Z, K, C, α, δ)
    walras = (Z * lag(K) ^ α - C) + (1 - δ) * lag(K) - K
    return walras
end

@simple function shocks(Z, ρ)
    ε = log(Z) - ρ * log(lag(Z))
    return ε
end

# make an RBC model and solve it's steady state
rbc_model = solve(
    model(households, firms, shocks),
    (C=1.0, K=1.0, Z=1.0),
    (γ=1.00, α=0.30, δ=0.25, β=(1/1.05), ρ=0.80)
)

# solve to the first order perturbation around the steady state
P, Q = solve(rbc_model, [:K, :C, :Z], [:ε], order=1);

## ESTIMATION ##############################################################################

# # for plotting MCMC chains with Makie.jl instead of Plots.jl
# include("../utilities/mcmc_plots.jl");

# rbc_model = RationalExpectationsModel(
#     [productivity_process, budget_dynamics, euler_equation], analytical_steady_state, [:ε]
# );

# rbc_ssm(; kwargs...) = state_space(
#     rbc_model, (; θ..., kwargs...), [:c], 1; algo=QuadraticIteration()
# );

# rng = MersenneTwister(1234);
# true_model = rbc_ssm();
# x, y = sample(rng, true_model, 100);

# @model function rbc_estimation(data)
#     # shock parameters
#     ρ ~ Uniform(-1.00, 1.00)
#     σ ~ InverseGamma(0.10, 2.00)

#     # model parameters
#     α ~ truncated(Normal(0.30, 0.15), 0.1, 0.8)
#     γ ~ Normal(0.40, 0.30)

#     # run the Kalman filter
#     _, logZ = GeneralisedFilters.filter(rbc_ssm(; ρ, σ, α, γ), KF(), data)
#     Turing.@addlogprob! logZ
# end;

# # this can be a little finnicky at times, so you may need to run it again
# chain = sample(rbc_estimation(y), NUTS(), MCMCThreads(), 2_000, 3);
# plot(chain, collect(θ[chain.name_map[:parameters]]); size=(900, 900))
