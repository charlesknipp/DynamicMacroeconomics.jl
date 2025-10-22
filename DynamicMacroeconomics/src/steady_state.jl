export solve

# evaluate F(x) given ... = x[t-1] = x[t] = x[t+1] =...
function steady_state(model::AbstractBlock, x::NamedTuple, θ::NamedTuple)
    variables = merge(θ, x)
    return merge(variables, model(variables))
end

function compute_residuals(
    model::AbstractBlock, x::NamedTuple, θ::NamedTuple, targets::NamedTuple{names}
) where {names}
    ss = steady_state(model, x, θ)
    return collect(map(i -> ss[i] - targets[i], names))
end

"""
    solve(model, initial_guess, calibration[, algorithm])

Solve for the steady state of a given graphical model given a calibration and initial guess,
uisng an optionally specified algorithm ala NonlinearSolve.jl (defaults to NewtonRaphson
with ForwardDiff)
"""
function solve(
    model::AbstractBlock,
    calibration::NamedTuple,
    initial_guess::NamedTuple{names},
    targets;
    algorithm = NewtonRaphson(; autodiff=AutoForwardDiff()),
    kwargs...
) where {names}
    # define the problem akin to the SciML ecosystem
    problem = NonlinearProblem(
        (x, θ) -> compute_residuals(model, NamedTuple{names}(x), θ, targets),
        collect(initial_guess),
        calibration
    )

    # solve with a simple Newton Ralphson algorithm
    solution = NonlinearSolve.solve(problem, algorithm; kwargs...)
    if SciMLBase.successful_retcode(solution)
        return steady_state(model, NamedTuple{names}(solution.u), calibration)
    else
        # not sure what to return here in the case of failure
        return solution
    end
end
