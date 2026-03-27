using DynamicMacroeconomics

@simple function euler_equation(y, πs, i, ωd, γ)
    euler_res = y - lead(y) + (1 / γ) * (i - lead(πs)) - ωd
    return euler_res
end

@simple function phillips_curve(y, πs, ωs, θ, β, φ, γ)
    nkpc_res = πs - (β * lead(πs) + (1 - θ) * (1 - θ * β) / θ * (γ + φ) * y - ωs)
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

θ = (
    β=0.995,
    θ=0.75,
    φ=1.00,
    ϕy=0.10,
    ϕπ=1.50,
    ϕi=0.90,
    γ=1.00,
    ρd=0.80,
    ρs=0.90,
    ρm=0.20,
    σd=1.6013,
    σs=0.9488,
    σm=0.2290,
);

teq_model = model(euler_equation, phillips_curve, taylor, ar_shocks; name="teq")
ss = solve(
    teq_model,
    (θ..., εs=0, εd=0, εm=0),
    (y=0, πs=0, i=0, ωs=0, ωd=0, ωm=0),
    (euler_res=0, nkpc_res=0, taylor_res=0, sres=0, dres=0, mres=0),
)

𝒥 = jacobian(
    teq_model,
    ss,
    (:y, :πs, :i, :ωs, :ωd, :ωm, :εs, :εd, :εm),
    (:euler_res, :nkpc_res, :taylor_res, :sres, :dres, :mres),
)

# get the VAR form as follows:
sys = FirstOrderSystem(𝒥, (:εs, :εd, :εm))
A, B = solve(sys, QZ())
