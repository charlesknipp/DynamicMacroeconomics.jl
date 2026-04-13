using ChainsMakie, CairoMakie

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
    fig = Figure(size=(600, size(mean_arr, 1) * 200))
    
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
    
    fig = Figure(size=(600, dims * 200))

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