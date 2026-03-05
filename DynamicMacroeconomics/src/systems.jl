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
function FirstOrderSystem(J::SparseJacobian{NU,NT}, shocks) where {NU,NT}
    NZ = length(shocks)
    ∂U = zeros(NT, NU - NZ, 3)
    ∂Z = zeros(NT, NZ, 3)
    for (o, outvar) in enumerate(outputs(J))
        for (i, invar) in enumerate(setdiff(inputs(J), shocks))
            ∂U[o, i, :] = J[outvar][invar]
        end
        for (i, invar) in enumerate(shocks)
            ∂Z[o, i, :] = J[outvar][invar]
        end
    end
    return FirstOrderSystem(∂U, ∂Z)
end

function FirstOrderSystem(model::CombinedBlock, ss, unknowns, shocks, targets; kwargs...)
    J = jacobian(model, ss, tuple(union(unknowns, shocks)...), targets; kwargs...)
    return FirstOrderSystem(J, shocks)
end
