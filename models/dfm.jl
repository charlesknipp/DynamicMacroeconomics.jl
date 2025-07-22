using LinearAlgebra, MatrixEquations, OffsetArrays
using SSMProblems, GeneralisedFilters
using Distributions
using Turing

const GF = GeneralisedFilters

## PREFACE #################################################################################

#=
This code is entirely generalizable, therefore it supports any state space model defined in
the SSMProblems interface. I use my specific branch of which for type consistent initial
state priors.

Therefore the user can define custom objects for both LatentDynamics and ObservationProcess,
and set the return type of a DPPL model to the SSM. This allows the generalized algorithms
to operate entirely within the Turing ecosystem.

NOTE: there is some additional overhead with this approach since the probablistic model
contains domain transformation dependent on operations performed on the stochastic nodes. I
have been working on a module that does this generalization without the overhead, but it is
still nascant and not yet ready for this type of analysis.

BABUR: focus on the block defined on line 124, this is where the state space is defined and
where most of your work will be relevant.
=#

## PLOTS ###################################################################################

# for plotting MCMC chains with Makie.jl instead of Plots.jl
include("../utilities/mcmc_plots.jl");

get_means(vals) = hcat(getproperty.(vals, Ref(:μ))...)

function plot_kalman_smoother(states)
    # collect only the mean from the smoother
    gaussian_means = cat(
        [cat(get_means.(states[:, i])..., dims=3) for i in axes(states, 2)]..., dims=4
    )

    # nx × T × chain_length × num_chains
    mean_arr = mean(gaussian_means, dims=3)
    fig = Figure()
    
    for i in axes(mean_arr, 1)
        ax = Axis(fig[1, i], title="Kalman Smoother")
        lines!.(Ref(ax), eachslice(mean_arr[i, :, 1, :], dims=2))
    end
    return fig
end

# NOTE: this only works for 1 dimensional states...
function plot_direct_iteration(chain)
    # TODO: update the naming convention to order by var then time index
    state_chain = group(chain, :x)
    
    fig = Figure()
    ax = Axis(fig[1, 1], title="Direct Iteration")
    
    for i in 1:size(chain, 3)
        mean_arrs = mean(state_chain, append_chains=false)[i][:, 2]
        lines!(ax, mean_arrs)
    end
    return fig
end

## KALMAN SMOOTHER #########################################################################

# modified kalman smoother from GF that stores the entire history
function smoother(
    model::GF.LinearGaussianStateSpaceModel,
    algo::KalmanSmoother,
    observations::AbstractVector;
    kwargs...,
)
    cache = GF.StateCallback(nothing, nothing)
    filtered, logZ = GF.filter(
        model, KF(), observations; callback=cache, kwargs...
    )

    rng = Random.default_rng()
    back_state = filtered
    smoothed_cache = fill(deepcopy(back_state), length(observations))

    for t in (length(observations) - 1):-1:1
        back_state = GF.backward(
            rng, model, algo, t, back_state, observations[t]; states_cache=cache, kwargs...
        )
        smoothed_cache[t] = back_state
    end

    return smoothed_cache, logZ
end

## POSTERIOR SOLVERS #######################################################################

@model function direct_iteration(state_space, data)
    ssm ~ to_submodel(state_space, false)
    x0 ~ SSMProblems.distribution(ssm.prior)
    x = OffsetVector(fill(x0, length(data) + 1), -1)
    for t in eachindex(data)
        x[t] ~ SSMProblems.distribution(ssm.dyn, t, x[t-1])
        data[t] ~ SSMProblems.distribution(ssm.obs, t, x[t])
    end
end

@model function marginalization(state_space, data)
    ssm ~ to_submodel(state_space, false)
    _, logZ = GeneralisedFilters.filter(ssm, KF(), data)
    Turing.@addlogprob! logZ
end

@model function smooth_marginalization(state_space, data)
    ssm ~ to_submodel(state_space, false)
    states, logZ = smoother(ssm, KalmanSmoother(), data)
    Turing.@addlogprob! logZ
    return states
end

## DFM EXAMPLE #############################################################################

# in the original code Babur originally set R = Diagonal([0.3, 0.4, 0.5, 0.4, 0.6]) and
# λs = [0.8, -0.3, 0.6, 1.2], where Λ = [1; λs]

@model function dynamic_factor_model(ny::Int)
    # random variables defined with a ~ operator
    λs ~ MvNormal(0.1I(ny - 1))
    σ  ~ Beta()

    # transition process is mean reverting random walk
    A = [0.85;;]
    Q = 0.4I(1)

    # factor loading normalized such that Λ[1] = 1 and measurement noise is iid
    Λ = reshape([1; λs], ny, 1)
    Σ = Diagonal(σ * ones(ny))

    # return the homogeneous linear Gaussian state space model
    return SSMProblems.StateSpaceModel(
        GF.HomogeneousGaussianPrior(zeros(1), lyapd(A, Q)),
        GF.HomogeneousLinearGaussianLatentDynamics(A, zeros(1), Q),
        GF.HomogeneousLinearGaussianObservationProcess(Λ, zeros(ny), Σ)
    )
end

## BENCHMARKS ##############################################################################

# define the baseline model (suppose we know σ)
state_space = dynamic_factor_model(10) | (; σ = 0.2)

# simulate from a provided vector of factor loadings
true_λs = randn(9)
true_model = state_space | (; λs = true_λs)
_, ys = sample(true_model(), 250)

# 947.20 seconds
chain_1 = sample(direct_iteration(state_space, ys), NUTS(), MCMCThreads(), 500, 3);
plot(group(chain_1, :λs), true_λs; size=(600, 1200))

# 91.98 seconds
chain_2 = sample(marginalization(state_space, ys), NUTS(), MCMCThreads(), 500, 3);
plot(group(chain_2, :λs), true_λs; size=(600, 1200))

## COMPARE STATES ##########################################################################

# plot direct iteration
plot_direct_iteration(chain_1)

# plot kalman smoother
states = returned(smooth_marginalization(state_space, ys), chain_2);
plot_kalman_smoother(states)
