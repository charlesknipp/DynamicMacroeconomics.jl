using DynamicMacroeconomics

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

## JACOBIAN DICTS ##########################################################################

# jacobians of simple blocks are sparse by default
𝒥1 = jacobian(households, ss, (:C, :K, :Z))
𝒥2 = jacobian(firms, ss, (:C, :K, :Z))
𝒥3 = jacobian(shocks, ss, (:Z, ))
