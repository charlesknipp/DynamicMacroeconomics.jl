# DynamicMacroeconomics

[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://charlesknipp.github.io/DynamicMacroeconomics.jl/)

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

We begin by defining the model equations, where each of the function arguments are state variables (including shocks).

```julia
@block function productivity(z, ε)
    z[t] = ρ * z[t-1] + σ * ε[t]
end

@block function euler(c, k, z)
    (c[t] ^ -γ) = (c[t+1] ^ -γ) * β * (α * exp(z[t+1]) * k[t] ^ (α - 1) + (1 - δ))
end

@block function budget(c, k, z)
    k[t] = (exp(z[t]) * k[t-1]^α - c[t]) + (1 - δ) * k[t-1]
end
```

As of now, the steady state must be determined by a user defined method.

```julia
function steady_state(θ)
    (; β, α, δ) = θ
    kss = ((1 / β - 1 + δ) / α) ^ (1 / (α - 1))
    css = kss ^ α - δ * kss
    return (c=css, k=kss, z=0)
end
```

Now that the model is sufficiently defined, we can obtain the policy function by solving the first order perturbation either by QZ decomposition (work in progress) or quadratic iteration.

```julia
rbc_model = RationalExpectationsModel([productivity, euler, budget], steady_state, [:ε])
A1, B1 = solve(rbc_model, 1; algo=QZ()) # broken
A2, B2 = solve(rbc_model, 1; algo=QuadraticIteration())
```

We encourage the user to experiment with `SSMProblems.jl` to create a potentially nonlinear measurement, but we include a constructor for linear Gaussian state space models. To demonstrate we can create a state space observing consumption with a measurement noise of 1.0, and simulate 100 time periods.

```julia
rng = MersenneTwister(1234)
ssm = state_space(rbc_model, θ, [:c], 1; algo=QuadraticIteration())
x, y = sample(rng, ssm, 100)
```

Using `GeneralizedFilters.jl`, we can extract the loglikelihood with the Kalman filter. This is used in `models/rbc.jl` for estimation. More details can be provided in that script.

## FAQ

- Models only support one lead and one lag as of now.
- I plan on adding the numerical calculation of steady states, but it is absent for now.
- QZ does not work for most models, since it was made prior to the macro which records timings.

Just ask me for questions on the implementation, and if you see anything questionable feel free to raise an issue or a PR.