using DynamicMacroeconomics
using Turing, Distributions, Random
using GeneralisedFilters

# for plotting MCMC chains with Makie.jl instead of Plots.jl
include("../utilities/mcmc_plots.jl");

@block function productivity_process(z, ε)
    z[t] = ρ * z[t-1] + σ * ε[t]
end;

@block function euler_equation(c, k, z)
    (c[t] ^ -γ) = (c[t+1] ^ -γ) * β * (α * exp(z[t+1]) * k[t] ^ (α - 1) + (1 - δ))
end;

@block function budget_dynamics(c, k, z)
    k[t] = (exp(z[t]) * k[t-1]^α - c[t]) + (1 - δ) * k[t-1]
end;

# solve the steady state analytically
function analytical_steady_state(θ)
    (; β, α, δ) = θ
    kss = ((1 / β - 1 + δ) / α) ^ (1 / (α - 1))
    css = kss ^ α - δ * kss
    return (c=css, k=kss, z=0)
end;

## ESTIMATION ##############################################################################

θ = (
    β = 1/1.05,
    α = 0.30,
    δ = 0.25,
    γ = 1.00,
    σ = 1.00,
    ρ = 0.80
);

rbc_model = RationalExpectationsModel(
    [productivity_process, budget_dynamics, euler_equation], analytical_steady_state, [:ε]
);

rbc_ssm(; kwargs...) = state_space(
    rbc_model, (; θ..., kwargs...), [:c], 1; algo=QuadraticIteration()
);

rng = MersenneTwister(1234);
true_model = rbc_ssm();
x, y = sample(rng, true_model, 100);

@model function rbc_estimation(data)
    # shock parameters
    ρ ~ Uniform(-1.00, 1.00)
    σ ~ InverseGamma(0.10, 2.00)

    # model parameters
    α ~ truncated(Normal(0.30, 0.15), 0.1, 0.8)
    γ ~ Normal(0.40, 0.30)

    # run the Kalman filter
    _, logZ = GeneralisedFilters.filter(rbc_ssm(; ρ, σ, α, γ), KF(), data)
    Turing.@addlogprob! logZ
end;

# this can be a little finnicky at times, so you may need to run it again
chain = sample(rbc_estimation(y), NUTS(), MCMCThreads(), 2_000, 3);
plot(chain, collect(θ[chain.name_map[:parameters]]); size=(900, 900))
