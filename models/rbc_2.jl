using DynamicMacroeconomics

@simple function households(K, L, w, φ, δ, γ, σ)
    C = (w / φ / L^(1 / σ))^γ
    I = K - (1 - δ) * lag(K)
    return C, I
end

@simple function firms(K, L, Z, α, δ)
    r = α * Z * (lag(K) / L)^(α - 1) - δ
    w = (1 - α) * Z * (lag(K) / L)^α
    Y = Z * lag(K)^α * L^(α - 1)
    return r, w, Y
end

@simple function market_clearing(C, I, Y, r, β, γ)
    goods_mkt = Y - C - I
    euler = C^(-1 / γ) - β * (1 + lead(r)) * lead(C)^(-1 / γ)
    return goods_mkt, euler
end

# calibrate instead to a target real interest rate
rbc_model = model(households, firms, market_clearing; name="rbc")
ss = solve(
    rbc_model,
    (L=1.00, σ=1.00, γ=1.00, δ=0.025, α=0.11),
    (φ=0.90, β=0.99, K=2.00, Z=1.00),
    (goods_mkt=0.00, r=0.01, euler=0.00, Y=1.00),
)

# the full system Jacobian is accessible using a custom sparse chain rule accumulation
𝒥 = jacobian(rbc_model, ss, (:K, :L, :Z), (:euler, :goods_mkt))
FirstOrderSystem(𝒥, (:Z,))

# alternatively you can obtain the first order system in one line
FirstOrderSystem(rbc_model, ss, (:K, :L), (:Z,), (:euler, :goods_mkt))
