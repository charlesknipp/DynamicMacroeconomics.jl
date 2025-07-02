module DynamicMacroeconomics

using Distributions
using Random

using GeneralisedFilters
using SSMProblems
using MatrixEquations
using LinearAlgebra

using DifferentiationInterface
import ForwardDiff

using Reexport
using MacroTools

@reexport using SSMProblems, GeneralisedFilters

export RationalExpectationsModel, steady_state, solve, @block, state_space

struct Block
    params::Set{Symbol}
    states::Set{Symbol}
    offsets::Dict{Int,Set{Symbol}}
    dynamics::Expr
end

macro block(args...)
    block_dict = splitdef(args[end])

    # get states and parameters
    params = symwalk(block_dict[:body])
    states = Set{Symbol}(block_dict[:args])

    # define a block object with the function name
    offsets, dynamics = parse_block(block_dict[:body])
    assignment = Expr(:(=), block_dict[:name], Block(params, states, offsets, dynamics))
    return esc(assignment)
end

function parse_block(expr::Expr)
    offsets = Dict{Int,Set{Symbol}}([
        (-1, Set{Symbol}()), (0, Set{Symbol}()), (1, Set{Symbol}())
    ])

    # capture timings for use in perturbation methods
    MacroTools.postwalk(expr) do ex
        if @capture(ex, var_[idx_])
            id = @eval $(Expr(:block, :(local t=0), ex.args[2]))
            offsets[id] = push!(offsets[id], var)
            return var
        else
            return ex
        end
    end

    # convert from y = f(x) to 0 = y - f(x)
    clean_block = MacroTools.postwalk(expr) do ex
        if @capture(ex, lhs_ = rhs_)
            return Expr(:call, :(-), lhs, rhs)
        else
            return ex
        end
    end

    # remove LineNumberNodes and stack into an array
    clean_expr = MacroTools.striplines(clean_block)
    stacked_expr = if clean_expr.head == :block
        Expr(:vcat, clean_expr.args...)
    else
        Expr(:vcat, clean_expr)
    end

    return offsets, stacked_expr
end

function collect_blocks(blocks)
    stacked_blocks = Expr[]
    offsets = Dict{Int,Set{Symbol}}([
        (-1, Set{Symbol}()), (0, Set{Symbol}()), (1, Set{Symbol}())
    ])

    for block in blocks
        update_timings!(offsets, block.offsets)
        push!(stacked_blocks, block.dynamics.args...)
    end

    params = union(getproperty.(blocks, :params)...)
    states = union(getproperty.(blocks, :states)...)

    return Block(params, states, offsets, Expr(:vcat, stacked_blocks...))
end

symwalk!(list) = ex -> begin
    if ex isa Symbol
        # include any other problematic symbols
        ex != :I && push!(list, ex)
    end

    if ex isa Expr
        if ex.head == :call || ex.head == :macrocall
            map(symwalk!(list), ex.args[2:end])
        elseif ex.head == :ref
            return list
        else
            map(symwalk!(list), ex.args)
        end
    end

    return list
end

symwalk(ex::Expr) = ex |> symwalk!(Set{Symbol}())

function update_timings!(main_dict, sub_dict)
    main_dict[-1] = union(main_dict[-1], sub_dict[-1])
    main_dict[0]  = union(main_dict[0], sub_dict[0])
    main_dict[1]  = union(main_dict[1], sub_dict[1])
    return main_dict
end

"""
    RationalExpectationsModel

A collection of model equations which represent a dynamic stochastic general equilibrium
model, where forward guidance is determined by information known up to the present.

Note: only models with a known analytical steady state are considered.

```julia
model = RationalExpectationsModel(
    [block_1, block_2, block_3],
    [:ε_1, :ε_2],
    steady_state_function
)
```

See also [`steady_state`](@ref).
"""
struct RationalExpectationsModel{F,SST,NX,NE}
    f::F
    states::NTuple{NX,Symbol}
    shocks::NTuple{NE,Symbol}
    offsets::Dict{Int64,Set{Symbol}}
    steady_state::SST
end

function (model::RationalExpectationsModel)(states, shocks, parameters, t::Int=2)
    return model.f(states, shocks, parameters, t)
end

function (model::RationalExpectationsModel)(parameters)
    steady_state = repeat(model.steady_state(parameters), 1, 3)
    nil_shocks = zeros(length(model.shocks), 3)
    return model(steady_state, nil_shocks, parameters)
end

function RationalExpectationsModel(
    blocks::Vector{Block}, steady_state::ST, shock_vars
) where {ST<:Function}
    combined_block = collect_blocks(blocks)
    state_vars = setdiff(combined_block.states, shock_vars)
    conditions = combined_block.dynamics

    steady_state_op = (θ) -> begin
        ss = steady_state(θ)
        T = Base.promote_eltypeof(ss...)
        collect(T, ss[[state_vars...]])
    end

    unpacked_params = Expr(:tuple, Expr(:parameters, combined_block.params...))
    unpacked_states = Expr(:tuple, state_vars...)
    unpacked_shocks = Expr(:tuple, shock_vars...)

    dynamic_op = @eval (states, shocks, θ, t) -> begin
        $(unpacked_params) = θ
        $(unpacked_states) = eachrow(states)
        $(unpacked_shocks) = eachrow(shocks)
        $(conditions)
    end

    states = tuple(state_vars...)
    shocks = tuple(shock_vars...)

    return RationalExpectationsModel(
        dynamic_op, states, shocks, combined_block.offsets, steady_state_op
    )
end

"""
    steady_state(model::RationalExpectationsModel, parameters)

Compute the model's steady state, where the return abides by the ordering determined in the
tuple containing the states `model.states`.

Currently, this module only supports analytical steady states. Although this is under heavy
consideration.

See also [`RationalExpectationsModel`](@ref).
"""
function steady_state(model::RationalExpectationsModel, parameters)
    return model.steady_state(parameters)
end

include("misc.jl")
include("state_space.jl")
include("perturbation_solutions.jl")

end # module DynamicMacroeconomics
