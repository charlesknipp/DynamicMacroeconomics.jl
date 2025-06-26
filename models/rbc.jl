using DynamicMacroeconomics
using Turing, Distributions, Random
using GeneralisedFilters
using StatsPlots

Base.@kwdef struct Parameters
    β = (1 / 1.05)
    ν = 0.80
    α = 0.30
    δ = 0.25
    γ = 1.00
    σ = 1.00
end

struct RBC <: RationalExpectationsModel
    parameters::Parameters
    function RBC(; kwargs...)
        return new(Parameters(; kwargs...))
    end
end

Base.size(::RBC) = (3, 1)

function DynamicMacroeconomics.optimality_conditions(model::RBC, y, ε, t::Int)
    (; β, ν, α, δ, γ, σ) = model.parameters
    z, k, c = y
    return [
        z[t] - ν * z[t-1] - σ * ε[];
        k[t] - (exp(z[t-1]) * k[t-1] ^ α - c[t]) - (1 - δ) * k[t-1];
        (c[t] ^ -γ) - (c[t+1] ^ -γ) * β * (α * exp(z[t]) * k[t] ^ (α - 1) + (1 - δ))
    ]
end

function DynamicMacroeconomics.steady_state(model::RBC)
    (; β, α, δ) = model.parameters
    kss = ((1 / β - 1 + δ) / α) ^ (1 / (α - 1))
    css = kss ^ α - δ * kss
    return [0; kss; css]
end

## ESTIMATION ##############################################################################

# TODO: move this to DynamicMacroeconomics
function state_space(model, varobs, ση²; kwargs...)
    A, B = solve(model, 1; kwargs...)
    (nx, nε), ny = size(model), length(varobs)
    C = I(nx)[varobs, :]
    return StateSpaceModel(
        GeneralisedFilters.HomogeneousGaussianPrior(zeros(nx), I(nx)),
        LinearGaussianControllableDynamics(A, B),
        GeneralisedFilters.HomogeneousLinearGaussianObservationProcess(
            C, zeros(ny), ση² * I(ny)
        )
    )
end

rng = MersenneTwister(1234);
θ = RBC();

true_model = solve_model(θ, [3], 1.0);
x, y = sample(rng, true_model, 100);

@model function rbc_model(data)
    # shock parameters
    ν ~ Uniform(-1.00, 1.00)
    σ ~ InverseGamma(0.10, 2.00)

    # model parameters
    α ~ truncated(Normal(0.30, 0.15), 0.1, 0.8)
    γ ~ Normal(0.40, 0.30)

    # observe consumption with unit variance and solve with the iterative approach
    model = state_space(
        RBC(; α, ν, γ, σ), [3], 1.0; algo=QuadraticIteration()
    )

    # run the Kalman filter
    _, logZ = GeneralisedFilters.filter(model, KF(), data)
    Turing.@addlogprob! logZ
end

# this can be a little finnicky at times, so you may need to run it again
chain = sample(rbc_model(y), NUTS(), 2_000);
plot(chain)
