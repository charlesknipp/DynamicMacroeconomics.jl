# DynamicMacroeconomics

[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://charlesknipp.github.io/DynamicMacroeconomics.jl/)

A general purpose DSGE interface which presents a lightweight approach to solving macroeconomic models. The purpose of creating this module was to rapidly estimate models in liu of (Childers, 2022).

As a departure to `DifferentiableStateSpaceModels.jl`, this module is entirely self contained and builds off of existing tools like `GeneralisedFilters.jl` for marginalized likelihood estimation.

## Installation

Upon cloning this repository, we have to properly set up the environment to support the local execution of this interface; furthermore, we also rely on a development branch of `GeneralisedFilters.jl` for functional automatic differentiation.

To set this up, instantiate the environment described in [Project.toml](https://github.com/charlesknipp/DynamicMacroeconomics.jl/blob/main/Project.toml), by running `Using Pkg; Pkg.activate("."); Pkg.instantiate()` if not activated automatically.

## A Simple Demonstration

First load in the module as well as some helpful packages.

```julia
using DynamicMacroeconomics
using GeneralisedFilters
using Random
```

We begin by defining the model equations, where each of the function arguments are state variables (including shocks).

```julia
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
```

Steady states are calculated internally, using `NonlinearSolve.jl` as a backend for optimization.

```julia
rbc_model = solve(
    model(households, firms, shocks),
    (C=1.0, K=1.0, Z=1.0),
    (γ=1.00, α=0.30, δ=0.25, β=(1/1.05), ρ=0.80)
)
```

Now that the model is sufficiently defined, we can obtain the policy function by solving the first order perturbation either by QZ decomposition (work in progress) or quadratic iteration.

```julia
A1, B1 = solve(rbc_model, [:K, :C, :Z], [:ε], 1; algo=QZ()) # broken
A2, B2 = solve(rbc_model, [:K, :C, :Z], [:ε], 1; algo=QuadraticIteration())
```

We encourage the user to experiment with `SSMProblems.jl` to create a potentially nonlinear measurement, but we include a constructor for linear Gaussian state space models. To demonstrate we can create a state space observing consumption with a measurement noise of 1.0, and simulate 100 time periods.

```julia
ssm = StateSpaceModel(..., LinearGaussianControllableDynamics(A2, B2), ...)
x, y = sample(rng, ssm, 100)
```

Using `GeneralizedFilters.jl`, we can extract the loglikelihood with the Kalman filter. This is used in `models/rbc.jl` for estimation. More details can be provided in that script.

## FAQ

- Models only support one lead and one lag as of now.
- I plan on adding the numerical calculation of steady states, but it is absent for now.
- QZ does not work for most models, since it was made prior to the macro which records timings.

Just ask me for questions on the implementation, and if you see anything questionable feel free to raise an issue or a PR.