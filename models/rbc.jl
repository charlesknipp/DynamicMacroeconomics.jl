using DynamicMacroeconomics
using CairoMakie

@simple function households(C, K, Z, α, β, γ, δ)
    euler = (C ^ -γ) - (β * (α * lead(Z) * K ^ (α - 1) + 1 - δ) * (lead(C) ^ -γ))
    return euler
end

@simple function firms(Z, K, C, α, δ)
    walras = (Z * lag(K) ^ α - C) + (1 - δ) * lag(K) - K
    return walras
end

@simple function shocks(Z, ρ, ε)
    shock_res = log(Z) - ρ * log(lag(Z)) - ε
    return shock_res
end

# make an RBC model and solve its steady state
rbc_model = model(households, firms, shocks, name="rbc")
ss = solve(
    rbc_model,
    (γ=1.00, α=0.30, δ=0.25, β=(1/1.05), ρ=0.80, ε=0.00),
    (C=1.00, K=0.40, Z=0.40),
    (euler=0.00, walras=0.00, shock_res=0.00)
)

# solve the first order perturbation to get the implied VAR (broken)
P, Q = solve(rbc_model, [:K, :C, :Z], [:ε], order=1, algo=QuadraticIteration());

# we can also solve for the sequence Jacobian and extract IRFs
G = solve(rbc_model, [:K, :C, :Z], [:ε], order=1, algo=SequenceJacobian(150));

## IMPULSE RESPONSES #######################################################################

function plot_jacobians(jacobians::AbstractArray{<:Real,3}; n::Int=5, dist::Int=5)
    theme = Theme(
        palette=(color=cgrad(:viridis, n+1, categorical=true),),
        Lines=(cycle=[:color], linewidth=2.5)
    )

    irfs = eachslice(jacobians, dims=1)
    with_theme(theme) do
        fig = Figure(size=(600, 300*size(jacobians, 1)))
        for i in eachindex(irfs)
            ax = Axis(fig[i,1], limits=((1, 50), nothing))
            for shock in 1:dist:(dist*n)
                lines!(ax, irfs[i][:, shock])
            end
        end
        return fig
    end
end

plot_jacobians(G; n=10, dist=3)
