using DynamicMacroeconomics
using FFTW
using LinearAlgebra
using MatrixEquations
using Tullio

## STRUCTURAL IDENTIFICATION ###############################################################

struct StateSpaceForm{T,MT<:AbstractMatrix{T}}
    A::MT
    B::MT
    C::MT
    D::MT
end

Base.size(ssm::StateSpaceForm) = (size(ssm.A, 1), size(ssm.C, 1), size(ssm.B, 2))
Base.size(ssm::StateSpaceForm, n::Integer) = size(ssm)[n]

function innovations_form(ssm::StateSpaceForm)
    Q, R = ssm.B * ssm.B', ssm.D * ssm.D'
    riccati, _, _ = ared(ssm.A', ssm.C', R, Q, ssm.B * ssm.D')
    K = (A * riccati * C' + B * D') * inv(C * riccati * C' + R)
    Σ = C * riccati * C' + R
    return K, Σ
end

function innovations_form(ssm::StateSpaceForm)
    K, Σ = innovation_cov(ssm)
    return StateSpaceForm(
        [ssm.A zero(ssm.A); (K * ssm.C) (ssm.A - K * ssm.C)],
        [ssm.B; K * ssm.D],
        [ssm.C -ssm.C],
        Matrix(ssm.D)
    ), Σ
end

function impulse_response(ssm::StateSpaceForm{XT}, T::Integer) where {XT}
    irf = zeros(XT, size(ssm, 2), size(ssm, 3), T)
    irf[:, :, 1] = ssm.D
    for t in 1:(T-1)
        irf[:, :, t + 1] = ssm.C * (ssm.A ^ (t-1)) * ssm.B
    end
    return irf
end

function spectral_covariance(M::AbstractArray{MT,3}) where {MT}
    ny, ne, T = size(M)
    Mpad = zeros(eltype(M), ny, ne, 2 * T - 2)
    copyto!(view(Mpad, :, :, 1:T), M)
    dft = rfft(Mpad, 3)
    @tullio r[o1, o2, t] := conj(dft[o1, z, t]) * dft[o2, z, t]
    return irfft(r, size(Mpad, 3), 3)[:, :, 1:T]
end

function identification_weights(ssm::StateSpaceForm{XT}, T::Integer) where {XT}
    whitened_model, Σ = whiten(ssm)
    M = impulse_response(whitened_model, T)
    r = zeros(XT, size(whitened_model, 3), T)
    invΣ = inv(Σ)
    @tullio r[i, t] := M[j, i, t] * invΣ[j, k] * M[k, i, t]
    return r
end

## THREE EQUATION MODEL ####################################################################

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

@simple function ar_shocks(ωs, ωd, ωm, εs, εd, εm, ρs, ρd, ρm, σs, σd, σm)
    sres = ωs - ρs * lag(ωs) - σs * εs
    dres = ωd - ρd * lag(ωd) - σd * εd
    mres = ωm - ρm * lag(ωm) - σm * εm
    return sres, dres, mres
end

# rip the parameters from (Wolf, 2020)
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

# define the model equations and solve for the steady state
teq_model = model(euler_equation, phillips_curve, taylor, ar_shocks; name="teq")
ss = solve(
    teq_model,
    (θ..., εs=0, εd=0, εm=0),
    (y=0, πs=0, i=0, ωs=0, ωd=0, ωm=0),
    (euler_res=0, nkpc_res=0, taylor_res=0, sres=0, dres=0, mres=0),
)

# compute the Jacobian for use in perturbation
𝒥 = jacobian(
    teq_model,
    ss,
    (:y, :πs, :i, :ωs, :ωd, :ωm, :εs, :εd, :εm),
    (:euler_res, :nkpc_res, :taylor_res, :sres, :dres, :mres),
)

# solve for the policy function to the first order
sys = FirstOrderSystem(𝒥, (:εs, :εd, :εm))
P, Q = solve(sys, QZ())

# permute the dimensions to match Dynare
p = [4, 2, 5, 6, 3, 1]
perm_P = I(6)[[4, 2, 5, 6, 3, 1], :] * P' * I(6)[:, p]
perm_Q = I(3)[[2, 1, 3], :] * Q' * I(6)[:, p]

# create the observable/canonical state space model
ssm = StateSpaceForm(
    perm_P[1:4, 1:4]', perm_Q[1:3, 1:4]', perm_P[1:4, [5, 6, 1]]', perm_Q[:, [5, 6, 1]]'
)

# we can test out the impulse responses and covariance functions like so:
M = impulse_response(ssm, 300)
Σ = spectral_covariance(M)

# for structural identification, compute the R^2 weights
identification_weights(ssm, 100)
