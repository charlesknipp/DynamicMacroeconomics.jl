export FirstOrderSystem

"""
    FirstOrderSystem

A first order approximation of a given rational expectations model which stores the state
jacobians `∂U` and the shock jacobian `∂Z`.
"""
struct FirstOrderSystem{T,ZT<:AbstractArray{T,3},UT<:AbstractArray{T,3}}
    ∂U::UT
    ∂Z::ZT
end

# TODO: this may need some work for abstractions like sequence space Jacobians
function FirstOrderSystem(J::BlockJacobian{T}, shocks) where {T}
    NT, NU, NZ = length(outputs(J)), length(inputs(J)), length(shocks)
    ∂U = zeros(T, NT, NU - NZ, 3)
    ∂Z = zeros(T, NT, NZ, 3)
    for (o, outvar) in enumerate(outputs(J))
        for (i, invar) in enumerate(setdiff(inputs(J), shocks))
            ∂U[o, i, :] = J[outvar, invar][-1:1]
        end
        for (i, invar) in enumerate(shocks)
            ∂Z[o, i, :] = J[outvar, invar][-1:1]
        end
    end
    return FirstOrderSystem(∂U, ∂Z)
end

function FirstOrderSystem(model::CombinedBlock, ss, unknowns, shocks, targets; kwargs...)
    J = jacobian(model, ss, tuple(union(unknowns, shocks)...), targets; kwargs...)
    return FirstOrderSystem(J, shocks)
end

# (unused) allows support for higher order leads and lags in QZ solver
function qz_helper(system::FirstOrderSystem)
    A, ∂Us = eachslice(system.∂U, dims=3)
    n = size(A, 1)
    m = length(∂Us) - 1
    Γ0 = [zeros(n * m, n) I(n * m); -hcat(∂Us...)]
    Γ1 = cat(I(n * m), A, dims=(1, 2))
    return Γ0, Γ1
end

# lag and lead operator as a ToeplitzSymbol
shift(::Type{T}, i::Integer) where {T} = ToeplitzSymbol(Dict(i => 1), T[1])
shift(i::Integer) = shift(Float64, i)
