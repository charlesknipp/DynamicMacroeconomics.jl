using DynamicMacroeconomics
using DifferentiationInterface
import ForwardDiff

Base.@kwdef struct Parameters
    β = (1 / 1.05)
    θ  = 0.75
    φ  = 1.00
    ϕy = 0.10
    ϕπ = 1.50
    ϕi = 0.90
    γ  = 1.00
end

"""
    TEQ(shock; kwargs...)

Three equation RANK model from (Leeper & Leith, 2016) with user defined exogenous shocks.
"""
struct TEQ{ST} <: RationalExpectationsModel
    parameters::Parameters
    shocks::ST
    function TEQ(shocks::ST; kwargs...) where {ST}
        return new{ST}(Parameters(; kwargs...), shocks)
    end
end

# replace once I get dispatch working for model construction via blocks
function base_optimality_conditions(model::TEQ, x, t::Int)
    (; β, θ, φ, ϕy, ϕπ, ϕi, γ) = model.parameters
    y, i, πs = x
    return [
        y[t] - y[t+1] + (1 / γ) * (i[t] - πs[t-1]);
        πs[t] - β * πs[t+1] - (1 - θ) * (1 - θ * β) / θ * (γ + φ) * y[t];
        i[t] - ϕi * i[t-1] - (1 - ϕi) * (ϕπ * πs[t] + ϕy * y[t])
    ]
end

function DynamicMacroeconomics.steady_state(model::TEQ)
    nx, _ = size(model)
    return zeros(nx)
end

## ONE TIME SHOCKS #########################################################################

struct MITShocks end

function DynamicMacroeconomics.optimality_conditions(model::TEQ{MITShocks}, x, ε, t::Int)
    y, i, πs = x
    εd, εs, εm = ε
    F = base_optimality_conditions(model, [y, i, πs], t) + [εd; -εs; εm]
    return F
end

function DynamicMacroeconomics.steady_state(::TEQ{MITShocks})
    return zeros(3)
end

Base.size(::TEQ{MITShocks}) = (3, 3)

## AR SHOCKS ###############################################################################

Base.@kwdef struct AR1Shocks
    ρd = 0.80
    ρs = 0.90
    ρm = 0.20
    σd = 1.60
    σs = 0.95
    σm = 0.25
end

function DynamicMacroeconomics.optimality_conditions(model::TEQ{AR1Shocks}, x, ε, t::Int)
    (; ρd, ρs, ρm, σd, σs, σm) = model.shocks
    y, i, πs, ωd, ωs, ωm = x
    εd, εs, εm = ε
    F = base_optimality_conditions(model, [y, i, πs], t) + [ωd[t]; -ωs[t]; ωm[t]]
    return [
        F;
        ωd[t] - ρd * ωd[t-1] - σd * εd;
        ωs[t] - ρs * ωs[t-1] - σs * εs;
        ωm[t] - ρm * ωm[t-1] - σm * εm
    ]
end

Base.size(::TEQ{AR1Shocks}) = (6, 3)

## DEMO ####################################################################################

# start with one time shocks
mit_shocks = MITShocks();
P1, Q1 = solve(
    TEQ(mit_shocks), 1; algo=QuadraticIteration(), backend=AutoForwardDiff()
)

# now add on the AR(1) shocks
ar1_shocks = AR1Shocks();
P2, Q2 = solve(
    TEQ(ar1_shocks; β=0.995), 1; algo=QuadraticIteration(), backend=AutoForwardDiff()
)
