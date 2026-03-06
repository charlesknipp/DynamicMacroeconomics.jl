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

We begin by defining the model equations, where each of the function arguments are state variables (including shocks). The desired syntax subscribes to the DAG paradigm ala SSJ, which implies variable substitution across blocks without a symbolic backend.

```julia
@simple function market_clearing(C, Y, I, r, β, γ)
    euler = (C^-γ) - (β * (1 + lead(r)) * (lead(C)^-γ))
    goods_mkt = Y - C - I
    return euler, goods_mkt
end

@simple function firms(Z, K, α, δ)
    r = α * Z * lag(K)^(α - 1) - δ
    Y = Z * lag(K)^α
    return r, Y
end

@simple function households(K, δ)
    I = K - (1 - δ) * lag(K)
    return I
end

@simple function shocks(Z, ρ, ε)
    shock_res = log(Z) - ρ * log(lag(Z)) - ε
    return shock_res
end
```

Steady states are calculated internally, using `NonlinearSolve.jl` as a backend for optimization.

```julia
rbc_model = model(market_clearing, firms, households, shocks; name="rbc")
ss = solve(
    rbc_model,
    (γ=1.00, α=0.30, δ=0.25, β=(1 / 1.05), ρ=0.80, ε=0.00),
    (C=1.00, K=0.40, Z=0.40),
    (euler=0.00, goods_mkt=0.00, shock_res=0.00),
)
```

Now that the model is sufficiently defined, we can obtain the policy function by solving the first order perturbation either by state space or sequence space methods.

```julia
A, B = solve(rbc_model, ss, [:K, :C, :Z], [:ε]; order=1, algo=QuadraticIteration())
GE   = solve(rbc_model, ss, [:K, :C, :Z], [:ε]; order=1, algo=SequenceJacobian(150))
```

We encourage the user to experiment with `SSMProblems.jl` to create a potentially nonlinear measurement, but we include a constructor for linear Gaussian state space models. To demonstrate we can create a state space observing consumption with a measurement noise of 1.0, and simulate 100 time periods.

```julia
ssm = StateSpaceModel(..., LinearGaussianControllableDynamics(A, B), ...)
x, y = sample(rng, ssm, 100)
```

Using `GeneralizedFilters.jl`, we can extract the loglikelihood with the Kalman filter. This is used in `models/rbc.jl` for estimation. More details can be provided in that script.

## Closing Remarks

- Models only support one lead and one lag as of now.
- Solving the first order perturbation is not guaranteed to work right now.
- I plan on removing the `ComponentArrays.jl` dependency since it may have some odd type conversions.
- Yes, I know the docs are broken and it will get fixed as soon as I can solve models consistently.

Just ask me for questions on the implementation, and if you see anything questionable feel free to raise an issue or a PR.