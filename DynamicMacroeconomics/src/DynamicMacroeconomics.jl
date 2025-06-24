module DynamicMacroeconomics

using Distributions, Random
using GeneralisedFilters, SSMProblems
using MatrixEquations, LinearAlgebra
using TaylorSeries
using Reexport

@reexport using SSMProblems, GeneralisedFilters
@reexport using TaylorSeries: set_variables

export RationalExpectationsModel

abstract type RationalExpectationsModel end

"""
    set_variables(model::RationalExpectationsModel)

label the series for which the model function takes in; this is only necessary for using
TaylorSeries.jl to calculate approximations.
"""
function TaylorSeries.set_variables(p::RationalExpectationsModel)
    throw(MethodError(set_variables, p))
end

"""
    model(yp, y, ym, model::RationalExpectationsModel)

Define the optimality conditions of the model; where yp is forward looking, y is
contemporaneous, and ym is lagged.

This is subject to change when I get around to making a nicer interface.
"""
function model(yp, y, ym, p::RationalExpectationsModel)
    throw(MethodError(model, (yp, y, ym, p)))
end


"""
    steady_state(model::RationalExpectationsModel)

Compute the model's steady state, where the return is ordered consistently with `model` and
`set_variables`.
"""
function steady_state(p::RationalExpectationsModel)
    throw(MethodError(steady_state, p))
end

"""
    construct_shock(model::RationalExpectationsModel; kwargs...)

Define the shock variances for a given model. As of now, this package only supports linear
and additive stochastic shocks like how ones defines an SDE in `DifferentialEquations.jl`.
"""
function construct_shock(p::RationalExpectationsModel; kwargs...)
    throw(MethodError(construct_shock, p))
end

include("misc.jl")
include("state_space.jl")
include("perturbation_solutions.jl")

end # module DynamicMacroeconomics
