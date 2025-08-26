export solve

struct SteadyStateModel{N}
    base_model::GraphicalModel{N}
    steady_state::NamedTuple
end

function Base.show(io::IO, model::SteadyStateModel)
    print(io, "SteadyStateModel:")
    for block in model.base_model.blocks
        inputs = show_variables(block.inputs)
        outputs = show_variables(block.outputs)
        print(io, "\n ($inputs) → ($outputs)")
    end
end

# evaluate F(x) given x[0] = x[1] = ... = x[t] = ...
function steady_state(model::GraphicalModel{N}, x, θ) where {N}
    variables = merge(θ, x)
    for i in 1:N
        invars = inputs(model[i])
        out = model[i](collect(variables[invars])...)
        variables = merge(variables, out)
    end
    return variables
end

"""
    solve(model, initial_guess, calibration[, algorithm])

Solve for the steady state of a given graphical model given a calibration and initial guess,
uisng an optionally specified algorithm ala NonlinearSolve.jl (defaults to NewtonRaphson
with ForwardDiff)
"""
function solve(
    model::GraphicalModel,
    initial_guess::NamedTuple{names},
    calibration;
    algorithm = NewtonRaphson(; autodiff=AutoForwardDiff()),
    kwargs...
) where {names}
    # define the problem akin to the SciML ecosystem
    problem = NonlinearProblem(
        (x, p) -> collect(steady_state(model, NamedTuple{names}(x), p)[model.targets]),
        collect(initial_guess),
        calibration
    )

    # solve with a simple Newton Ralphson algorithm
    solution = NonlinearSolve.solve(problem, algorithm; kwargs...)
    if SciMLBase.successful_retcode(solution)
        ss = steady_state(model, NamedTuple{names}(solution.u), calibration)
        return SteadyStateModel(model, ss[setdiff(keys(ss), model.targets)])
    else
        # not sure what to return here in the case of failure
        return nothing
    end
end
