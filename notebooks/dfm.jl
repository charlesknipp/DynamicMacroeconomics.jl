### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 464b489a-6642-11f0-2c92-df60ccaeff24
# ╠═╡ show_logs = false
begin
	import Pkg
	Pkg.activate(mktempdir())
	Pkg.instantiate()
	
	# add SSMProblems GeneralisedFilters
	Pkg.add([
	    Pkg.PackageSpec(name="SSMProblems", rev="ck/priors"),
	    Pkg.PackageSpec(name="GeneralisedFilters", rev="ck/priors")
	])

	# add the rest
	Pkg.add(["MatrixEquations", "OffsetArrays", "Distributions", "Turing", "CairoMakie", "MCMCChains"])

	using LinearAlgebra, MatrixEquations, OffsetArrays
	using SSMProblems, GeneralisedFilters
	using Distributions
	using Turing
	using CairoMakie
	using MCMCChains
end

# ╔═╡ 77aff319-8b90-410f-ab9c-823c0d6d1c07
begin
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
	        Density = (
				cycle=[:strokecolor=>:patchcolor],
				strokewidth=1,
				color=:transparent
			),
	        Lines = (
				linewidth=1,
			)
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
end

# ╔═╡ c29239dd-c5df-49f2-a1b9-b7f6e33b50be
md"""
The dynamic factor model in question is a simple linear Gaussian state space model with one state and N observables.

$x_{t} = \rho x_{t-1} + \varepsilon_{t} \quad \quad
y_{t} = \begin{bmatrix} 1 \\ \lambda_{1} \\ \vdots \\ \lambda_{N} \end{bmatrix} x_{t} + \eta_{t}$

where $\eta \sim N(0, \Sigma_{\eta})$ and $\varepsilon \sim N(0, \Sigma_{\varepsilon})$.

There are appropriate restrictions on $\rho$ such that the transition process is stationary, and we suppose that both covariances are known.

Using `SSMProblems` and `GeneralisedFilters` we can define this state space model with the function `create_homogeneous_linear_gaussian_model`, which constructs an SSM with a linear Gaussian prior, transition, and observation process. This constructor is fully type stable, and allows for automatic differentiation.

It should be noted that automatic differentiation of `lyapd` is undefined in this instance, so we limit estimation to the factors $\lambda$.
"""

# ╔═╡ 26a6ca39-c85c-4863-99b6-f6f9868cdae9
function dynamic_factor_model(loadings::Vector{ΛT}, σ::ΣT) where {ΛT<:Real, ΣT<:Real}
    ny = length(loadings) + 1
    nx = 1

    A = [0.85;;]
    Q = Diagonal([0.4])

    Λ = reshape([1; loadings], ny, 1)
    Σ = Diagonal(σ * ones(ny))

    return create_homogeneous_linear_gaussian_model(
        zeros(ΛT, 1), lyapd(A, Q), A, zeros(ΛT, nx), Q, Λ, zeros(ΣT, ny), Σ
    )
end

# ╔═╡ 190bba96-939a-4de9-9794-9a5486c2d2cd
@model function dfm_state_space(ny::Int)
	λs ~ MvNormal(0.1I(ny - 1))
	return dynamic_factor_model(λs, 0.2)
end

# ╔═╡ 44a8bed5-61e0-414a-b460-69155964eb70
begin
	true_λs = randn(9)
	true_model = dynamic_factor_model(true_λs, 0.2)
	_, ys = sample(true_model, 250)
end;

# ╔═╡ 1e8c7d28-e577-41c1-892a-49f4c3337be2
md"""
## Marginalization

The standard approach for calculating likelihoods of state space models where parameters are sampled and states are marginalized. In a typical linear Gaussian setting, this is fast, efficient, and standard in most macroeconomic models which are linearized to a first order perturbation.
"""

# ╔═╡ cdf52f2e-3e0c-4017-9e79-504e66f06f7a
@model function marginalization(data)
	ssm ~ to_submodel(dfm_state_space(length(data[1])), false)

    # run a filtering algorithm (here we use the Kalman filter)
    _, logZ = GeneralisedFilters.filter(ssm, KF(), data)
    Turing.@addlogprob! logZ
end

# ╔═╡ 45761a4d-e679-4719-9bf7-cfcf2bf4135d
kf_chains = sample(marginalization(ys), NUTS(), MCMCThreads(), 500, 3);

# ╔═╡ e87888c7-2c0e-45d3-b069-6a37cb1e1f27
plot(group(kf_chains, :λs), true_λs; size=(600, 1200))

# ╔═╡ 783b52fe-8f0a-4046-9b6d-91086ff62c95
md"""
## Direct Iteration

The approach taken by (Childers et al, 2022) relies instead on joint estimation of the hidden states and parameters without marginalizing the likelihood calculation.
"""

# ╔═╡ 07bde636-6db3-441f-a0da-eb2e61e14503
@model function direct_iteration(data)
    ssm ~ to_submodel(dfm_state_space(length(data[1])), false)

    # sample from the state space model directly
    x0 ~ SSMProblems.distribution(ssm.prior)
    x = OffsetVector(fill(x0, length(data) + 1), -1)
    for t in eachindex(data)
        x[t] ~ SSMProblems.distribution(ssm.dyn, t, x[t-1])
        data[t] ~ SSMProblems.distribution(ssm.obs, t, x[t])
    end
end

# ╔═╡ 947d42f4-9200-4e89-be1c-b9a531bc9704
di_chains = sample(direct_iteration(ys), NUTS(), MCMCThreads(), 500, 3);

# ╔═╡ d41323ad-24f0-4cc1-963d-5fb2f053f56c
plot(group(di_chains, :λs), true_λs; size=(600, 1200))

# ╔═╡ Cell order:
# ╟─464b489a-6642-11f0-2c92-df60ccaeff24
# ╟─77aff319-8b90-410f-ab9c-823c0d6d1c07
# ╟─c29239dd-c5df-49f2-a1b9-b7f6e33b50be
# ╠═26a6ca39-c85c-4863-99b6-f6f9868cdae9
# ╠═190bba96-939a-4de9-9794-9a5486c2d2cd
# ╠═44a8bed5-61e0-414a-b460-69155964eb70
# ╟─1e8c7d28-e577-41c1-892a-49f4c3337be2
# ╠═cdf52f2e-3e0c-4017-9e79-504e66f06f7a
# ╠═45761a4d-e679-4719-9bf7-cfcf2bf4135d
# ╠═e87888c7-2c0e-45d3-b069-6a37cb1e1f27
# ╟─783b52fe-8f0a-4046-9b6d-91086ff62c95
# ╠═07bde636-6db3-441f-a0da-eb2e61e14503
# ╠═947d42f4-9200-4e89-be1c-b9a531bc9704
# ╠═d41323ad-24f0-4cc1-963d-5fb2f053f56c
