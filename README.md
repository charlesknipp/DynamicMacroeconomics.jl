# DynamicMacroeconomics

A general purpose DSGE interface which presents a lightweight approach to solving macroeconomic models. The purpose of creating this module was to rapidly estimate models in liu of (Childers, 2022).

As a departure to `DifferentiableStateSpaceModels.jl`, this module is entirely self contained and builds off of existing tools like `GeneralisedFilters.jl` for marginalized likelihood estimation.

## Installation

Upon cloning this repository, we have to properly set up the environment to support the local execution of this interface; furthermore, we also rely on a development branch of `GeneralisedFilters.jl` for functional automatic differentiation.

To set this up, simply run the script `julia setup.jl` in your terminal. This should only need to occur once until I get this PR finally merged.

## A Simple Demonstration

First load in the module as well as some helpful packages.

```julia
using DynamicMacroeconomics
using GeneralisedFilters
using Random
```

We begin by defining an RBC model by it's parameters and keeping track of the model variables (in our case this is a vector of TFP, capital, and consumption)

```julia
Base.@kwdef struct Parameters
    β = (1 / 1.05)
    ν = 0.80
    α = 0.30
    δ = 0.25
    γ = 1.00
    σ = 1.00
end

Base.size(::RBC) = (3, 1)
```

Where optimality conditions represent the models transition dynamics.

```julia
function DynamicMacroeconomics.optimality_conditions(model::RBC, y, ε, t::Int)
    (; β, ν, α, δ, γ, σ) = model.parameters
    z, k, c = y
    return [
        z[t] - ν * z[t-1] - σ * ε[];
        k[t] - (exp(z[t]) * k[t-1]^α - c[t]) - (1 - δ) * k[t-1];
        (c[t] ^ -γ) - (c[t+1] ^ -γ) * β * (α * exp(z[t+1]) * k[t] ^ (α - 1) + (1 - δ))
    ]
end
```

As of now, the steady state must be determined by a user defined method.

```julia
function DynamicMacroeconomics.steady_state(model::RBC)
    (; β, α, δ) = model.parameters
    kss = ((1 / β - 1 + δ) / α) ^ (1 / (α - 1))
    css = kss ^ α - δ * kss
    return [0; kss; css]
end
```

Now that the model is sufficiently defined, we can obtain the policy function by solving the first order perturbation either by QZ decomposition or quadratic iteration.

```julia
rbc_model = RBC()
A1, B1 = solve(rbc_model, 1; algo=QZ())
A2, B2 = solve(rbc_model, 1; algo=QuadraticIteration())
```

We encourage the user to experiment with `SSMProblems.jl` to create a potentially nonlinear measurement, but we include a constructor for linear Gaussian state space models. To demonstrate we can create a state space observing consumption with a measurement noise of 1.0 and shock variance of 1.4, and simulate 100 time periods.

```julia
rng = MersenneTwister(1234)
ssm = StateSpaceModel(model, [3], 1.0; algo=QZ())
x, y = sample(rng, ssm, 100)
```

Using `GeneralizedFilters.jl`, we can extract the loglikelihood with the Kalman filter. This is used in `models/rbc.jl` for estimation. More details can be provided in that script.

## FAQ

As of now, models are limited to a (Schmitt-Grohe & Uribe, 2004) paradigm. Essentially where forward looking variables are on different timings than state variables. There are ways around this, but my code isn't robust to deviations thus far.

Just ask me for questions on the implementation, and if you see anything questionable feel free to raise an issue or a PR.