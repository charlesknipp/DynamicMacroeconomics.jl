"""
    FirstOrderPerturbation

A first order approximation of a given rational expectations model which stores the state
jacobians `∂Y` and the shock jacobian `∂E`.
"""
struct FirstOrderPerturbation
    ∂Y::AbstractArray{<:Real,3}
    ∂E::AbstractArray{<:Real,2}
    function FirstOrderPerturbation(
        model::RationalExpectationsModel, parameters; backend=AutoForwardDiff()
    )
        ε = zeros(Bool, length(model.shocks))
        nil_shocks = [ε ε ε]
        ss = steady_state(model, parameters)

        ∂Y = cat(
            jacobian(x -> model([ss ss x], nil_shocks, parameters), backend, ss),
            jacobian(x -> model([ss x ss], nil_shocks, parameters), backend, ss),
            jacobian(x -> model([x ss ss], nil_shocks, parameters), backend, ss),
            dims = 3
        )
        ∂E = jacobian(x -> model([ss ss ss], [ε x ε], parameters), backend, ε)
        return new(∂Y, ∂E)
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

not_in(x, y::AbstractArray) = isempty(x) ? ones(Bool, length(y)) : @. !in(x, y)

function solve(system::FirstOrderPerturbation, ::QZ)
    # TODO: remove static variables with a QR decomp of B for their respective col
    A, B, C = eachslice(system.∂Y, dims=3)
    idx = findall.(x -> any(abs.(x) .> 0), eachcol.([A, C]))

    mixed_idx  = intersect(idx...)
    mixed_loc  = sort.(indexin.(Ref(mixed_idx), idx))
    strict_idx = symdiff.(idx, Ref(mixed_idx))
    not_mixed  = indexin.(strict_idx, idx)

    ns, nm = length.(idx), length(mixed_idx)
    reorder = [strict_idx[2]; idx[1]]

    # partition the system as in (Villemot, 2011)
    Γ012, Γ112 = getindex.([A, -B], Ref(:), Ref(idx[1]))
    Γ011, Γ111 = getindex.([B, -C], Ref(:), Ref(idx[2]))
    Γ011 = Γ011 * Diagonal(not_in(mixed_idx, idx[2]))

    # pad the remainder to enforce square matrices
    Γ122, Γ021 = getindex.(I.(ns), mixed_loc, Ref(:))
    Γ022, Γ121 = zeros.(Ref(Bool), Ref(nm), ns)

    F = schur([Γ011 Γ012; Γ021 Γ022], [Γ111 Γ112; Γ121 Γ122])
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

    ghx = [hx[not_mixed[2], :]; gx;;] * I(size(system.∂Y, 1))[idx[2], :]
    ghx = ghx[reorder, :]
    
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

function solve(model::RationalExpectationsModel, params, ::Val{1}; algo=QZ(), kwargs...)
    system = FirstOrderPerturbation(model, params; kwargs...)
    return solve(system, algo)
end
