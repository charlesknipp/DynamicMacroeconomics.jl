using DynamicMacroeconomics
using Turing, Distributions, Random
using GeneralisedFilters
using StatsPlots

# isoelastic utility of consumption
utility(c, γ) = γ == 1 ? log(c) : c^(1 - γ) / (1 - γ)
marginal_utility(c, γ) = c ^ (-γ)
intertemporal_elasticity(γ) = 1 / γ

# cobb-douglas production using technology and capital
output(z, k, α) = exp(z) * k ^ α
marginal_product(z, k, α) = α * exp(z) * k ^ (α - 1)

Base.@kwdef struct RBC <: RationalExpectationsModel
    ρ = 0.05
    ν = 0.80
    α = 0.30
    δ = 0.25
    γ = 1.00
end

# this substitution is equivalent to a market clearing condition Y = C + I
investment(z, k, c, p::RBC) = output(z, k, p.α) - c

# define our sequence of named variables
DynamicMacroeconomics.set_variables(::RBC) = set_variables([:z, :k, :c])

function DynamicMacroeconomics.model(yp, y, ym, p::RBC)
    # unpack parameters and convert the exponential discount factor
    (; ν, ρ, δ) = p
    β = 1 / (1 + ρ)

    _,  _,  cp = yp
    z,  k,  c  = y
    zm, km, _  = ym

    return [
        z - ν * zm,
        k - (investment(zm, km, c, p)) - (1 - δ) * km,
        marginal_utility(c, p.γ) - marginal_utility(cp, p.γ) * β * (marginal_product(z, k, p.α) + (1 - δ))
    ]
end

function DynamicMacroeconomics.steady_state(p::RBC)
    (; ρ, α, δ) = p
    kss = ((ρ + δ) / (α)) ^ (1 / (α - 1))
    css = kss ^ α - δ * kss
    return [0, kss, css]
end

DynamicMacroeconomics.construct_shock(::RBC; σz::Real=1, kwargs...) = [σz; 0; 0;;]

## ESTIMATION ##############################################################################

# define an SSM where we observe consumption with unit noise
function make_ssm(p::RBC, σz::Real; kwargs...)
    return StateSpaceModel(p, [3], 1.0; σz, tol=1e-8, kwargs...);
end

rng = MersenneTwister(1234);
θ = RBC();

true_model = rbc_ssm(θ, 1.0);
x, y = sample(rng, true_model, 100);

@model function rbc_model(data, method)
    # shock parameters
    ν  ~ Uniform(-1.00, 1.00)
    σz ~ InverseGamma(0.10, 2.00)

    # model parameters
    α ~ truncated(Normal(0.30, 0.15), 0.1, 0.8)
    γ ~ Normal(0.40, 0.30)

    # set depreciation to a constant
    θ = RBC(; α, ν, γ)
    ssm = make_ssm(θ, σz; method)

    # run the Kalman filter
    _, logZ = GeneralisedFilters.filter(ssm, KF(), data)
    Turing.@addlogprob! logZ
end

# this can be a little finnicky at times, so you may need to run it again
chain = sample(rbc_model(y, :iteration), NUTS(), MCMCThreads(), 1_000, 5);
plot(chain)
