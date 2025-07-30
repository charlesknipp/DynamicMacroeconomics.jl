using LinearAlgebra, MatrixEquations, OffsetArrays
using SSMProblems, GeneralisedFilters
using Distributions, Random
using Turing
using Serialization
using ChainsMakie, CairoMakie

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

## PLOTS ###################################################################################

function plot_chains(chain, observables; legend=false)
    # customize the theme a little bit
    mcmc_theme = Theme(
        ChainsDensity=(alpha=0.1, strokewidth=1.5,), TracePlot=(linewidth=1.5,)
    )

    # plot the chains using ChainsMakie.jl
    fig = with_theme(mcmc_theme) do
        plot(chain)
    end

    # add the true values
    ammended_theme = (color=:tomato, linewidth=1.5, linestyle=:dash)
    for (i, param) in enumerate(observables)
        hlines!(fig[i, 1], param; ammended_theme...)
        vlines!(fig[i, 2], param; ammended_theme...)
    end

    # remove the legend and trim the last row
    if !legend
        delete!(contents(fig[end,:])[1])
        trim!(fig.layout)
    end

    return fig
end

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
        ax = if i == 1
            Axis(fig[i, 1], title="Kalman Smoother")
        else
            Axis(fig[i, 1])
        end

        lines!.(ax, eachslice(mean_arr[i, :, 1, :], dims=2))
    end
    return fig
end

function plot_direct_iteration(chain, dims::Int)
    mean_arrs = mean(group(chain, :x), append_chains=false)
    
    fig = Figure()
    
    for j in 1:dims
        ax = if j == 1
            Axis(fig[j, 1], title = "Direct Iteration")
        else
            Axis(fig[j, 1])
        end

        for i in 1:size(chain, 3)
            states = reshape(mean_arrs[i][:, 2], dims, :)
            lines!(ax, states[j, :])
        end
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

    # transition process is a dampened spline smoother
    ϕ = @. (-1) ^ (1:nx) * binomial(nx, 1:nx)
    A = diagm(-1 => ones(nx - 1))
    A[1, :] .= -ϕ
    A .*= 0.85

    # unlike spline smoother add noise to identify mixed signals
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
state_space = dynamic_factor_model(5, 2) | (; σ = 0.2);

# simulate from a provided vector of factor loadings
true_λs = randn(num_factors(5, 2));
true_model = state_space | (; λs = true_λs);
_, ys = sample(true_model(), 250);

# 5106.79 seconds
chain_1 = sample(direct_iteration(state_space, ys), NUTS(), MCMCThreads(), 500, 3);
plot_chains(group(chain_1, :λs), true_λs)
serialize("data/joint_chain.jls", chain_1)

# 183.96 seconds
chain_2 = sample(marginalization(state_space, ys), NUTS(), MCMCThreads(), 500, 3);
plot_chains(group(chain_2, :λs), true_λs)
serialize("data/marginal_chain.jls", chain_2)

## COMPARE STATES ##########################################################################

# plot direct iteration
plot_direct_iteration(chain_1, 2)

# plot kalman smoother
states = returned(smooth_marginalization(state_space, ys), chain_2);
plot_kalman_smoother(states)
