"""
    FirstOrderPerturbation

A first order approximation of a given rational expectations model which stores the state
jacobians `∂Y` and the shock jacobian `∂E`.
"""
struct FirstOrderPerturbation{M<:RationalExpectationsModel}
    ∂Y::AbstractArray{<:Real,3}
    ∂E::AbstractArray{<:Real,2}
    function FirstOrderPerturbation(
        model::M; backend=AutoForwardDiff()
    ) where {M<:RationalExpectationsModel}
        ss, ε = steady_state(model), zeros(Bool, size(model)[2])
        ∂Y = jacobian_sequence(model, ss, ε; backend)
        ∂E = DifferentiationInterface.jacobian(x -> model([ss ss ss], x), backend, ε)
        return new{M}(∂Y, ∂E)
    end
end

"""
    QZ()

The algorithm borrowed from (Klein, 1999) for solving the forward looking expectations form
and converting the quadratic system into a reduced form linear Gaussian policy function.

Note: this is a non-differentiable algorithm and will cause automatic differentiation to
fail in the case of HMC, until a custom rule is made.

# Example

Given a rational expectations model `model`, we solve the first order system like so:

```julia
system = FirstOrderPerturbation(model);
policy, impact = solve(system, QZ());
```

See also [`QuadraticIteration`](@ref).
"""
struct QZ end

function solve(system::FirstOrderPerturbation, ::QZ)
    # TODO: remove static variables with a QR decomp of B for their respective col
    A, B, C = eachslice(system.∂Y, dims=3)
    idx = findall.(x -> any(abs.(x) .> 0), eachcol.([A, C]))

    mixed_idx  = intersect(idx[1], idx[2])
    strict_idx = sort.(setdiff.(idx, Ref(mixed_idx)))

    ns = length.(strict_idx)
    nm = length(mixed_idx)
    n = sum(ns) + nm

    # arrange as is specified in (Villemot, 2011)
    Γ0 = [B[:, idx[2]] A[:, strict_idx[1]]; zeros(Bool, nm, n - nm) I(nm)]
    Γ1 = [-C[:, strict_idx[2]] -B[:, strict_idx[1]]; I(nm) zeros(Bool, nm, n - nm)]

    F = schur(Γ0, Γ1)
    eigenvalues = F.β ./ F.α

    stable_flag = abs.(eigenvalues) .< 1
    nstable = count(stable_flag)
    ordschur!(F, stable_flag)

    z21 = F.Z[nstable+1:end, 1:nstable]
    z11 = F.Z[1:nstable, 1:nstable]

    s11 = F.S[1:nstable, 1:nstable]
    t11 = F.T[1:nstable, 1:nstable]

    if rank(z11) < nstable
        warning("Invertibility condition violated")
    end

    gx = z21 * inv(z11)
    hx = z11 * (s11 \ t11) * inv(z11)

    ghx = [hx; gx;;] * I(size(system.∂Y, 1))[idx[2], :]
    return ghx, (A * ghx + B) \ -system.∂E
end

"""
    QuadraticIteration(; tol=1e-12, max_iters=2^10)

A brute force algorithm which uses an iterative root finding scheme to converge to a policy
function. While this approach is much simpler than the QZ decomposition, it is considerably
slower and less accurate. Despite its flaws, it is fully differentiable via AD, and is the
correct choice for running HMC estimation.

# Example

Given a rational expectations model `model`, we solve the first order system like so:

```julia
system = FirstOrderPerturbation(model);
policy, impact = solve(system, QuadraticIteration());
```

See also [`QZ`](@ref).
"""
Base.@kwdef struct QuadraticIteration
    tol::Real = 1e-12
    max_iters::Int = 2^10
end

function solve(system::FirstOrderPerturbation, algo::QuadraticIteration)
    A, B, C = eachslice(system.∂Y, dims=3)
    ghx = zero(A)

    for _ in 1:algo.max_iters
        ghx = -(A*ghx + B) \ C
        if maximum(C + B*ghx + A*ghx*ghx) < algo.tol
            break
        end
    end

    return ghx, (A * ghx + B) \ -system.∂E
end

function solve(model::RationalExpectationsModel, ::Val{1}; algo=QZ(), kwargs...)
    system = FirstOrderPerturbation(model; kwargs...)
    return solve(system, algo)
end
