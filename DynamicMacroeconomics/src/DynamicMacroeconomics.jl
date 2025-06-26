module DynamicMacroeconomics

using Distributions, Random
using GeneralisedFilters, SSMProblems
using MatrixEquations, LinearAlgebra
using DifferentiationInterface
using Reexport

@reexport using SSMProblems, GeneralisedFilters

export RationalExpectationsModel, steady_state, optimality_conditions, solve

abstract type RationalExpectationsModel end

function (model::RationalExpectationsModel)(y::AbstractArray, ε::AbstractArray, t::Int=2)
    # not too crazy about this, but at least it's hidden from the user
    return optimality_conditions(model, eachrow(y), eachrow(ε), t)
end

function (model::RationalExpectationsModel)(y::AbstractArray, ε::AbstractVector, t::Int=2)
    # noise is almost always a vector, but there may be some special cases with ARMA shocks
    return optimality_conditions(model, eachrow(y), ε, t)
end

"""
    optimality_conditions(model::RationalExpectationsModel, y, ε, t)

The optimality conditions of `model` with states `y` and noise `ε` at time `t`. A required
component to a model's definition.

# Example

```julia
struct Demo <: RationalExpectationsModel end

function optimality_conditions(::Demo, y, ε, t)
    a, b = y
    εa, εb = ε
    return [
        a[t] - 0.1 * a[t-1] - εa,
        b[t] ^ 2.0 - b[t-1] - εb
    ]
end
```

See also [`steady_state`](@ref).
"""
function optimality_conditions(model::RationalExpectationsModel, y, ε, t)
    error("model equations not defined for $(typeof(model))")
end

"""
    steady_state(model::RationalExpectationsModel)

Compute the model's steady state, where the return is ordered consistently with `model` and
`set_variables`. Currently, the user must specify this function, but I plan to solve this
numerically at some point in the future.

See also [`optimality_conditions`](@ref).
"""
function steady_state(::M) where {M <: RationalExpectationsModel}
    error("steady state not defined for type $M")
end

function Base.size(::M) where {M <: RationalExpectationsModel}
    # TODO: remove the need for this function...
    error("the user must define `size` such that it returns a tuple of (ny, nε)")
end

include("misc.jl")
include("state_space.jl")
include("perturbation_solutions.jl")

end # module DynamicMacroeconomics
