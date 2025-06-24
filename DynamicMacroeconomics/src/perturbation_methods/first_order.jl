function linearize(p::RationalExpectationsModel)
    set_variables(p)
    ss = steady_state(p)
    return FirstOrderPerturbation((
        TaylorSeries.jacobian(taylor_expand(y -> model(y, ss, ss, p), ss, order=1), ss),
        TaylorSeries.jacobian(taylor_expand(y -> model(ss, y, ss, p), ss, order=1), ss),
        TaylorSeries.jacobian(taylor_expand(y -> model(ss, ss, y, p), ss, order=1), ss)
    ))
end

struct FirstOrderPerturbation{T<:AbstractArray}
    jacobians::NTuple{3,T}
    indicies::NTuple{2,Vector{Int}}
    function FirstOrderPerturbation(jacobians::NTuple{3,T}) where {T<:AbstractArray}
        us = findall(x -> any(abs.(x) .> 0), eachcol(jacobians[1]))
        xs = findall(x -> any(abs.(x) .> 0), eachcol(jacobians[3]))
        return new{T}(jacobians, (xs, us))
    end
end

function get_jacobians(model::FirstOrderPerturbation)
    _, xp, x = getindex.(model.jacobians, Ref(:), Ref(model.indicies[1]))
    up, u, _ = getindex.(model.jacobians, Ref(:), Ref(model.indicies[2]))
    return up, u, xp, x
end

# based on the method of (Klein, 1999) for ideas defined in (Blanchard-Kahn, 1980)
function qz(Γ0, Γ1)
    F = schur(-Γ0, Γ1)
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
    return hx, gx
end

# calculate the policy function and the shock response for the implied VAR form
function construct_system(
    model::FirstOrderPerturbation, shocks::AbstractArray, ::Val{:qz}; kwargs...
)
    nvars = sum(length, model.indicies)
    nx, _ = model.indicies

    up, u, xp, x = get_jacobians(model)
    hx, gx = qz([xp up], [x u])

    ghx = [hx; gx;;] * diagm(ones(Base.promote_eltype(hx, gx), nvars))[nx, :]
    ghu = [up * gx + xp u] \ shocks

    return ghx, ghu
end

function quadratic_iteration(A, B, C; tol=1e-16, maxiters=2^10, kwargs...)
    X, Y = zero(A), zero(C)
    for _ in 1:maxiters
        X = -(A*X + B) \ C
        Y = -(C*Y + B) \ A
        if maximum(C + B*X + A*X*X) < tol
            break
        end
    end

    # if maximum(abs.(eigvals(Y))) > 1.0
    #     error("No stable equilibrium")
    # end

    return X
end

# calculate the policy function and the shock response for the implied VAR form
function construct_system(
    model::FirstOrderPerturbation, shocks::AbstractArray, ::Val{:iteration}; kwargs...
)
    J = model.jacobians
    A = quadratic_iteration(J...)
    B = (J[1] * A + J[2]) \ shocks
    return A, B
end

function solve(p::RationalExpectationsModel, ::Val{1}; method=:qz, kwargs...)
    shocks = construct_shock(p; kwargs...)
    model = linearize(p)
    return construct_system(model, shocks, Val(method); kwargs...)
end
