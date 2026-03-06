using DynamicMacroeconomics

@simple function market_clearing(C, Y, I, r, β, γ)
    euler = (C^-γ) - (β * (1 + lead(r)) * (lead(C)^-γ))
    goods_mkt = Y - C - I
    return euler, goods_mkt
end

@simple function firms(Z, K, α, δ)
    r = α * Z * lag(K)^(α - 1) - δ
    Y = Z * lag(K)^α
    return r, Y
end

@simple function households(K, δ)
    I = K - (1 - δ) * lag(K)
    return I
end

@simple function shocks(Z, ρ, ε)
    shock_res = log(Z) - ρ * log(lag(Z)) - ε
    return shock_res
end

# make a traditional RBC model and solve its steady state
rbc_model = model(market_clearing, firms, households, shocks; name="rbc")
ss = solve(
    rbc_model,
    (γ=1.00, α=0.30, δ=0.25, β=(1 / 1.05), ρ=0.80, ε=0.00),
    (C=1.00, K=0.40, Z=0.40),
    (euler=0.00, goods_mkt=0.00, shock_res=0.00),
)

# the full system Jacobian is accessible using a custom sparse chain rule accumulation
𝒥 = jacobian(rbc_model, ss, (:C, :K, :Z, :ε), (:euler, :goods_mkt, :shock_res))
FirstOrderSystem(𝒥, (:ε,))

# alternatively you can obtain the first order system in one line
FirstOrderSystem(rbc_model, ss, (:C, :K, :Z), (:ε,), (:euler, :goods_mkt, :shock_res))
