using CairoMakie
using MCMCChains

function plot_density(chains::Chains, fig; label=false)
    params = chains.name_map[:parameters]
    _, _, n_chains = size(chains)

    for (i, param) in enumerate(params)
        ax = Axis(fig[i, 1]; ylabel=string(param))
        for chain in 1:n_chains
            values = chains[:, param, chain]
            density!(ax, values; label=string(chain))
        end
    
        hideydecorations!(ax; label=!label, grid=false)
    end
end

function plot_iterations(chains::Chains, fig)
    params = chains.name_map[:parameters]
    n_samples, _, n_chains = size(chains)

    for (i, param) in enumerate(params)
        ax = Axis(fig[i, 1]; ylabel=string(param))
        for chain in 1:n_chains
            values = chains[:, param, chain]
            lines!(ax, 1:n_samples, values; label=string(chain))
        end
    
        hideydecorations!(ax; label=false, grid=false)
    end
end

function plot(chains::Chains; kwargs...)
    mcmc_theme = Theme(
        Density = (cycle=[:strokecolor=>:patchcolor], strokewidth=1, color=:transparent),
        Lines = (linewidth=1,)
    )

    fig = Figure(; kwargs...)
    with_theme(mcmc_theme) do
        plot_iterations(chains, fig[1,1])
        plot_density(chains, fig[1,2])
    end

    return fig
end

function plot(chains::Chains, observed_data; kwargs...)
    fig = plot(chains; kwargs...)
    ammended_theme = (color=:tomato, linewidth=1.5, linestyle=:dash)

    for (i, param) in enumerate(observed_data)
        hlines!(fig[1, 1][i, 1], param; ammended_theme...)
        vlines!(fig[1, 2][i, 1], param; ammended_theme...)
    end

    return fig
end
