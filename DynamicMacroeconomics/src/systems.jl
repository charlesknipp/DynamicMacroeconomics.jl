export FirstOrderSystem

"""
    FirstOrderSystem

A first order approximation of a given rational expectations model which stores the state
jacobians `∂Z` and the shock jacobian `∂U`.
"""
struct FirstOrderSystem{T,ZT<:AbstractArray{T,3},UT<:AbstractArray{T,3}}
    ∂Z::ZT
    ∂U::UT
end

# TODO: this may need some work for abstractions like sequence space Jacobians
function FirstOrderSystem(J::SparseJacobian{NU,NT}, exogvars) where {NU,NT}
    NZ = length(exogvars)
    ∂U = zeros(NT, NU - NZ, 3)
    ∂Z = zeros(NT, NZ, 3)
    for (o, outvar) in enumerate(outputs(J))
        for (i, invar) in enumerate(setdiff(inputs(J), exogvars))
            ∂U[o, i, :] = J[outvar][invar]
        end
        for (i, invar) in enumerate(exogvars)
            ∂Z[o, i, :] = J[outvar][invar]
        end
    end
    return RecursiveSystem(Float64, ∂U, ∂Z)
end
