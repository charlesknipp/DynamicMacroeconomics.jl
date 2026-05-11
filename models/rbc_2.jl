using DynamicMacroeconomics
using CairoMakie
using ToeplitzMatrices

## PLOTTING FUNCTIONS ######################################################################

function plot_responses(A::Matrix, space::Integer=10; maxval::Integer=50)
    fig = Figure()
    ax = Axis(fig[1, 1]; limits=((0, maxval), nothing))
    for i in space:space:(maxval + 1)
        lines!(ax, 0:maxval, A[1:(maxval + 1), i]; color=(i / maxval), colorrange=0:1)
    end
    return fig
end

function make_policy(A, varnames, offset::Integer)
    B = BlockJacobian(eltype(A), varnames, varnames)
    make_policy!(B, A, offset)
    return B
end

function make_policy!(B::BlockJacobian{T}, A::AbstractMatrix{T}, offset::Integer) where {T}
    for (i, input) in enumerate(inputs(B)), (o, output) in enumerate(outputs(B))
        sym = ToeplitzSymbol(T)
        sym[offset] = A[o, i]
        B[output, input] += sym
    end
    return B
end

# TODO: make this general and add it to module
function state_space_form(A::BlockJacobian, controls)
    P, _ = solve(A, controls, QZ())

    states = symdiff(inputs(A), controls)
    ∂U = subset(A, :, states)
    ∂Z = subset(A, :, controls)

    Q = BlockJacobian(eltype(P), controls, states)
    P = make_policy(P, states, -1)

    # compute the matrix polynomial for the state space form
    system = DynamicMacroeconomics.matrix_polynomial(
        ∂U, getband(P, -1), DynamicMacroeconomics.offset_range(∂U)[2:end]
    )

    make_policy!(Q, system \ -getband(∂Z, 0), 0)
    make_policy!(Q, system \ -(getband(∂U, 1) * getband(Q, 0) + getband(∂Z, 1)), 1)

    return P, Q
end

## RBC MODEL ###############################################################################

@simple function households(K, L, w, φ, δ, γ, σ)
    C = (w / φ / L^(1 / σ))^γ
    I = K - (1 - δ) * lag(K)
    return C, I
end

@simple function firms(K, L, Z, α, δ)
    r = α * Z * (lag(K) / L)^(α - 1) - δ
    w = (1 - α) * Z * (lag(K) / L)^α
    Y = Z * lag(K)^α * L^(1 - α)
    return r, w, Y
end

@simple function market_clearing(C, I, Y, r, β, γ)
    goods_mkt = Y - C - I
    euler = C^(-1 / γ) - β * (1 + lead(r)) * lead(C)^(-1 / γ)
    return goods_mkt, euler
end

# calibrate instead to a target real interest rate
rbc_model = model(households, firms, market_clearing; name="rbc")
ss = solve_steady_state(
    rbc_model,
    (L=1.00, σ=1.00, γ=1.00, δ=0.025, α=0.11),
    (φ=0.90, β=0.99, K=2.00, Z=1.00),
    (goods_mkt=0.00, r=0.01, euler=0.00, Y=1.00),
)

# the full system Jacobian is accessible using a custom sparse chain rule accumulation
𝒥 = jacobian(rbc_model, ss, (:K, :L, :Z), (:euler, :goods_mkt))
G = solve(𝒥, (:Z,), SequenceJacobian(300))

# plot the sequence space Jacobians
plot_responses(G[:K, :Z], 5; maxval=50)
plot_responses(G[:L, :Z], 5; maxval=50)

# we can compare this to the state space form as well
P, Q = state_space_form(𝒥, (:Z,))

# initial shock at t = 0
init_shock = Q
irfs = zeros(2, 300)

# stack the contemporaneous effects in irfs
for t in 1:300
    irfs[1, t] = init_shock[:K, :Z][-t + 1]
    irfs[2, t] = init_shock[:L, :Z][-t + 1]
    init_shock = P * init_shock
end

# the following difference should be relatively close to zero
sum(irfs[1, :] .- G[:K, :Z, :, 1]) < 1e-10
