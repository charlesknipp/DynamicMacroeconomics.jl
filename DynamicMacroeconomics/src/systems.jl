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
    model::SteadyStateModel{N}, state_variables, shock_variables
) where {N}
    base_model, steady_state = model.base_model, model.steady_state
    partials = merge([get_partials(base_model[i], steady_state) for i in 1:N]...)
    system, shocks = [], []
    for target in base_model.targets
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
