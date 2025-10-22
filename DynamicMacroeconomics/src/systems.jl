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
    partials = get_partials(block, steady_state)
    system, shocks = [], []
    for target in targets
        ∂Zi = get.(Ref(partials[target]), state_variables, Ref(zeros(Bool, 3)))
        ∂Ui = get.(Ref(partials[target]), shock_variables, Ref(zeros(Bool, 3)))
        push!(system, cat(∂Zi..., dims=2))
        push!(shocks, cat(∂Ui..., dims=2))
    end

    return FirstOrderSystem(
        permutedims(cat(system..., dims=3), (3, 2, 1)),
        permutedims(cat(shocks..., dims=3), (3, 2, 1))
    )
end
