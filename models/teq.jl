using DynamicMacroeconomics

@simple function euler_equation(y, πs, i, ωs, γ)
    euler_res = y - lead(y) + (1 / γ) * (i - lag(πs)) - ωs
    return euler_res
end

@simple function phillips_curve(y, πs, ωd, θ, β, φ, γ)
    nkpc_res = πs - (β * lead(πs) + (1 - θ) * (1 - θ * β) / θ * (γ + φ) * y - ωd)
    return nkpc_res
end

@simple function taylor(y, πs, i, ωm, ϕi, ϕπ, ϕy)
    taylor_res = i - ϕi * lag(i) - (1 - ϕi) * (ϕπ * πs + ϕy * y) - ωm
    return taylor_res
end

@simple function ar_shocks(ωs, ωd, ωm, εs, εd, εm, ρs, ρd, ρm)
    sres = ωs - ρs * lag(ωs) - εs
    dres = ωd - ρd * lag(ωd) - εd
    mres = ωm - ρm * lag(ωm) - εm
    return sres, dres, mres
end

## DEMO ####################################################################################

θ1 = (
    β = (1 / 1.05),
    θ  = 0.75,
    φ  = 1.00,
    ϕy = 0.10,
    ϕπ = 1.50,
    ϕi = 0.90,
    γ  = 1.00
);

# start with one time shocks
teq_model_1 = solve(
    model(euler_equation, phillips_curve, taylor),
    (y=0, πs=0, i=0), (θ1..., ωm=0, ωs=0, ωd=0)
)

sys = FirstOrderSystem(teq_model_1, [:y, :πs, :i], [:ωs, :ωd, :ωm])
sys.∂Z

θ2 = (;
    θ1...,
    ρd = 0.80,
    ρs = 0.90,
    ρm = 0.20,
    σd = 1.60,
    σs = 0.95,
    σm = 0.25
);

# add AR(1) shocks
teq_model_2 = solve(
    model(euler_equation, phillips_curve, taylor, ar_shocks),
    (y=0, πs=0, i=0, ωs=0, ωd=0, ωm=0), (θ2..., εs=0, εd=0, εm=0)
)

sys = FirstOrderSystem(teq_model_2, [:y, :πs, :i, :ωs, :ωd, :ωm], [:εs, :εd, :εm])
sys.∂Z
