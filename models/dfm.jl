using LinearAlgebra, MatrixEquations, OffsetArrays
using SSMProblems, GeneralisedFilters
using Distributions, Random
using Turing
using Serialization
using Dates
using Logging, LoggingExtras

import AbstractMCMC, DynamicPPL
import ProgressLogging: ProgressString, ProgressLevel

# change the number of progress logger updates to send to the log file
AbstractMCMC.DEFAULT_N_UPDATES = 30

const GF = GeneralisedFilters

## PREFACE #################################################################################

#=
This code is entirely generalizable, therefore it supports any state space model defined in
the SSMProblems interface. I use my specific branch of which for type consistent initial
state priors.

Therefore the user can define custom objects for both LatentDynamics and ObservationProcess,
and set the return type of a DPPL model to the SSM. This allows the generalized algorithms
to operate entirely within the Turing ecosystem.

NOTE: there is a minor type stability issue with the direct iteration solver. It shouldn't
impact performance too much, but in case the type of the state space prior differs from the
transition dynamics, there may be a mild slow down. I will patch this one out.
=#

## LOGGING #################################################################################

parse_log(message::AbstractString) = (true, message)
function parse_log(message::ProgressString)
    if isnothing(message.progress.fraction)
        if message.progress.done
            return (true, "Finished")
        else
            return (false, "")
        end
    else
        return (true, message.progress)
    end
end

prog_logger = FormatLogger(open("logs.txt", "w")) do io, args
    level = (args.level == Logging.LogLevel(-1)) ? "Progress" : string(args.level)
    out, msg = parse_log(args.message)
    out && println(io, "[$level] $(msg)")
end

LOGGER = TeeLogger(
    MinLevelLogger(
        prog_logger,
        ProgressLevel
    ),
    MinLevelLogger(
        FileLogger("errors.txt"),
        Logging.Error
    )
)

## POSTERIOR SOLVERS #######################################################################

# this might be type unstable because of the offset vector filling the type of the prior
@model function direct_iteration(state_space, data)
    ssm ~ to_submodel(state_space, false)
    x0 ~ SSMProblems.distribution(ssm.prior)
    x = OffsetVector(fill(x0, length(data) + 1), -1)
    for t in eachindex(data)
        x[t] ~ SSMProblems.distribution(ssm.dyn, t, x[t-1])
        data[t] ~ SSMProblems.distribution(ssm.obs, t, x[t])
    end
end

# this is guaranteed to be type stable
@model function marginalization(state_space, data)
    ssm ~ to_submodel(state_space, false)
    _, logZ = GeneralisedFilters.filter(ssm, KF(), data)
    Turing.@addlogprob! logZ
end

# callback situation is dire, but stability is non-essential for collecting smooth states
@model function smooth_marginalization(state_space, data)
    ssm ~ to_submodel(state_space, false)
    states, logZ = smoother(ssm, KalmanSmoother(), data)
    Turing.@addlogprob! logZ
    return states
end

## DFM EXAMPLE #############################################################################

# very rudimentary, but it works for any sized matrix
function factor_matrix(λs::AbstractVector{T}, ny::Int, nx::Int) where {T}
    Λ = diagm(ny, nx, ones(T, min(nx, ny)))
    iter = 1
    for i in 1:ny, j in 1:nx
        if i > j
            Λ[i, j] = λs[iter]
            iter += 1
        end
    end
    return Λ
end

num_factors(ny::Int, nx::Int) = ny * nx - sum(1:nx)

@model function dynamic_factor_model(ny::Int, nx::Int)
    # random variables defined with a ~ operator
    λs ~ MvNormal(0.1I(num_factors(ny, nx)))
    σ  ~ Beta()

    # transition process is a dampened iid random walk
    A = 0.85I(nx)

    # add noise to identify mixed signals
    Q = 0.4I(nx)

    # factor loading normalized on the diagonals
    Λ = factor_matrix(λs, ny, nx)
    Σ = Diagonal(σ * ones(ny))

    # return the homogeneous linear Gaussian state space model
    return SSMProblems.StateSpaceModel(
        GF.HomogeneousGaussianPrior(zeros(nx), lyapd(A, Q)),
        GF.HomogeneousLinearGaussianLatentDynamics(A, zeros(nx), Q),
        GF.HomogeneousLinearGaussianObservationProcess(Λ, zeros(ny), Σ)
    )
end

## BENCHMARKS ##############################################################################

# define the baseline model (suppose we know σ)
state_space = dynamic_factor_model(3, 3) | (; σ = 0.2);

# simulate from a provided vector of factor loadings
rng = MersenneTwister(1234)
true_λs = randn(rng, num_factors(3, 3));
true_model = state_space | (; λs = true_λs);
_, _, ys = sample(true_model(), 250);

function run_sampler(model::DynamicPPL.Model)
    model_name = string(typeof(model).parameters[1])
    dims = model.args.state_space.args
    with_logger(LOGGER) do
        chain = sample(model, NUTS(), MCMCThreads(), 500, 3)
        serialize("data/$(Dates.today()) $(model_name) ($(dims.nx), $(dims.ny)).jls", chain)
    end
end

# direct iteration
run_sampler(direct_iteration(state_space, ys))

# marginalization
run_sampler(marginalization(state_space, ys))
