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

function FirstOrderSystem(
    block::AbstractBlock, steady_state, state_variables, shock_variables, targets
)
    error("needs work")
end
