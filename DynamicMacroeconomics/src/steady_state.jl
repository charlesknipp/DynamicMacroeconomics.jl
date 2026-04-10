export solve_steady_state

# for combining component arrays
Base.merge(x::ComponentVector, y::ComponentVector) = ComponentVector(; x..., y...)

# evaluate F(x) given ... = x[t-1] = x[t] = x[t+1] =...
function steady_state(block::AbstractBlock, x, θ)
    variables = merge(θ, x)
    return merge(variables, block(variables))
end

# for subtracting named tuples with shared name spaces
subtract(A::ComponentArray, B::ComponentArray) = A - B
function subtract(A::NamedTuple{Names}, B::NamedTuple{Names}) where {Names}
    return NamedTuple{Names}(values(A) .- values(B))
end

function compute_residuals(block::AbstractBlock, unknowns, calibration, targets)
    outputs = block(merge(unknowns, calibration))
    return collect(subtract(outputs[keys(targets)], targets))
end

"""
    solve_steady_state(model, initial_guess, calibration, targets[, algorithm])

Solve for the steady state of a given graphical model given a calibration and initial guess,
uisng an optionally specified algorithm ala NonlinearSolve.jl (defaults to NewtonRaphson
with ForwardDiff).

Users can get creative and solve for steady states that target certain model outputs; even
when said output is not targetted in the general equilibrium solution.

For details, see the example script [`rbc_2.jl`](@ref) which calibrates the model to find a
target rental rate of capital.
"""
function solve_steady_state(
    block::AbstractBlock,
    calibration::ComponentArray,
    unknowns::ComponentArray,
    targets::ComponentArray;
    algorithm=NewtonRaphson(; autodiff=AutoForwardDiff()),
    kwargs...,
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

function solve_steady_state(
    block::AbstractBlock,
    calibration::NamedTuple,
    unknowns::NamedTuple,
    targets::NamedTuple;
    kwargs...,
)
    return solve_steady_state(
        block,
        ComponentArray(calibration),
        ComponentArray(unknowns),
        ComponentArray(targets);
        kwargs...,
    )
end
