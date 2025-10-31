export solve

# for combining component arrays
Base.merge(x::ComponentVector, y::ComponentVector) = ComponentVector(; x..., y...)

# evaluate F(x) given ... = x[t-1] = x[t] = x[t+1] =...
function steady_state(block::AbstractBlock, x, θ)
    variables = merge(θ, x)
    return merge(variables, block(variables))
end

function compute_residuals(
    block::AbstractBlock,
    calibration::ComponentArray,
    unknowns::ComponentArray,
    targets::ComponentArray
)
    outputs = block(merge(unknowns, calibration))
    return outputs[keys(targets)] - targets
end

"""
    solve(model, initial_guess, calibration[, algorithm])

Solve for the steady state of a given graphical model given a calibration and initial guess,
uisng an optionally specified algorithm ala NonlinearSolve.jl (defaults to NewtonRaphson
with ForwardDiff)
"""
function solve(
    block::AbstractBlock,
    calibration::ComponentArray,
    unknowns::ComponentArray,
    targets::ComponentArray;
    algorithm = NewtonRaphson(; autodiff=AutoForwardDiff()),
    kwargs...
)
    # define the problem akin to the SciML ecosystem
    problem = NonlinearProblem(
        (x, θ) -> compute_residuals(block, x, θ, targets), unknowns, calibration
    )

    # solve with a simple Newton Ralphson algorithm
    solution = NonlinearSolve.solve(problem, algorithm; kwargs...)
    if SciMLBase.successful_retcode(solution)
        return steady_state(block, solution.u, calibration)
    else
        # not sure what to return here in the case of failure
        return solution
    end
end

function solve(
    block::AbstractBlock,
    calibration::NamedTuple,
    unknowns::NamedTuple,
    targets::NamedTuple;
    kwargs...
)
    return solve(
        block,
        ComponentArray(calibration),
        ComponentArray(unknowns),
        ComponentArray(targets);
        kwargs...
    )
end