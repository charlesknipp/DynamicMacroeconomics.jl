module DynamicMacroeconomics

using Distributions
using Random

using SSMProblems
using MatrixEquations
using LinearAlgebra
using ToeplitzMatrices

using SparseArrays

using DifferentiationInterface
using ForwardDiff: ForwardDiff
using ADTypes

using NonlinearSolve

using Reexport
using MacroTools
using Graphs
using ComponentArrays
using OffsetArrays
using DataStructures

export model, @simple, lead, lag, jacobian
export SimpleBlock, CombinedBlock, ComposedBlock
export inputs, outputs

# for steady state evaluations
offset(x::Real, ::Int) = x

# we must pass an offset array for time offsets
offset(x::OffsetVector, t::Int) = x[t]

abstract type AbstractBlock end

function Base.show(io::IO, b::AbstractBlock)
    inputs = join(b.inputs, ", ")
    outputs = join(b.outputs, ", ")
    return print(io, "$(b.name): ($inputs) → ($outputs)")
end

function inputs(::AbstractBlock) end
function outputs(::AbstractBlock) end

"""
    SimpleBlock{FT,SP}

A standard equation block which captures targets in the return statement and state variables
in the function arguments. To keep things as general as possible, function parameters are
initially treated as states. This allows us to calibrate parameters given a target output.

Model equations are subsumed into the functor attribute of each simple block, which are a
heavily modified version of the user specified code. Although, the macro will preserve any
LineNumberNodes, so any relevant stack traces will point back to the users original code.
"""
struct SimpleBlock{FT,SP} <: AbstractBlock
    functor::FT
    inputs::OrderedSet{Symbol}
    outputs::OrderedSet{Symbol}
    name::String
    sparsity::SP
end

inputs(block::SimpleBlock) = collect(block.inputs)
outputs(block::SimpleBlock) = collect(block.outputs)

macro simple(args...)
    func_def = MacroTools.splitdef(args[end])
    block_name = func_def[:name]
    inputs = OrderedSet{Symbol}(func_def[:args])

    # build an anonymous function call with gensym("functor")
    functor = gensym(func_def[:name])
    func_def[:name] = functor
    outputs = build_expression!(func_def)

    # store the block sparsity for efficient evaluation of the Jacobian
    sparse_io = sparsity_detector(func_def[:body], inputs, outputs)

    # clean up the assignment operators to ensure the scope is entirely local
    hygeine!(func_def)

    # eval the functor and define the simple block associated with it
    return MacroTools.@q begin
        $(MacroTools.combinedef(func_def))
        $(esc(block_name)) = $(SimpleBlock)(
            $(functor), $(inputs), $(outputs), $(string(block_name)), $(sparse_io)
        )
    end
end

function build_expression!(func_dict)
    targets = Symbol[]
    func_dict[:body] = MacroTools.postwalk(func_dict[:body]) do ex
        # changes the return structure to guarantee a NamedTuple
        if @capture(ex, return target_)
            if target isa Expr
                push!(targets, target.args...)
                return :(return (; $(target.args...)))
            else
                # return singular targets as a tuple
                push!(targets, target)
                return :(return (; $target,))
            end
        else
            return ex
        end
    end

    func_dict[:body] = MacroTools.postwalk(func_dict[:body]) do ex
        if ex isa Symbol && ex in func_dict[:args]
            return :(offset($ex, 0))
        else
            return ex
        end
    end

    func_dict[:body] = MacroTools.postwalk(func_dict[:body]) do ex
        if @capture(ex, f_(offset(var_, 0)))
            f == :lead && return :(offset($var, 1))
            f == :lag && return :(offset($var, -1))
        end
        return ex
    end

    return OrderedSet(targets)
end

# keep the assignment scope such that resolving from a named tuple is escaped
function hygeine!(func_dict)
    inputs = func_dict[:args]
    func_dict[:args] = [gensym()]
    func_dict[:body] = MacroTools.@q begin
        local (; $(map(esc, inputs)...)) = $(func_dict[:args]...)
        $(MacroTools.postwalk(x -> x in inputs ? esc(x) : x, func_dict[:body]))
    end
end

function sparsity_detector(body, inputs, outputs)
    # make relevant substitutions for proper index handling
    exprmap = Dict{Symbol,Expr}()
    MacroTools.postwalk(body) do ex
        if @capture(ex, LHS_ = RHS_) && (LHS isa Symbol)
            exprmap[LHS] = RHS
            (LHS in outputs) && return ex
        elseif (ex in keys(exprmap))
            return exprmap[ex]
        end
        return ex
    end

    # return a sorted set per each argument of populated indices
    offset_dict = Dict(arg => Set{Int64}() for arg in inputs)
    for out in outputs
        MacroTools.postwalk(exprmap[out]) do ex
            @capture(ex, offset(x_, t_)) && push!(offset_dict[x], t) 
            return ex
        end
    end

    return Dict(k => minimum(v):maximum(v) for (k, v) in offset_dict)
end

(block::SimpleBlock)(x::NamedTuple) = block.functor(x)
(block::SimpleBlock)(x::NamedTuple, c::NamedTuple) = block.functor(merge(c, x))

# TODO: totally forgot what this does but it looks type unstable
function (block::SimpleBlock)(var::Symbol, x::AbstractVector{<:Real}, c::NamedTuple)
    return block(NamedTuple{(var,)}((x,)), c)
end

# workaround for AD using OffsetArrays
offset_component(X::Real, offset_range) = X
offset_component(X::AbstractVector, offset_range) = OffsetVector(X, offset_range)

# NOTE: work around since component arrays does not support offset arrays
function (block::SimpleBlock)(x::ComponentArray)
    out = block.functor(
        (; (i => offset_component(x[i], block.sparsity[i]) for i in inputs(block))...)
    )
    return ComponentVector(out)
end

# when taking partials given a constant (not type stable)
(block::SimpleBlock)(x::ComponentArray, c::ComponentArray) = block([x; c])

"""
    SolvedBlock{BT}

A self contained model with a solvable component. Taking some simple block, an unknown, and
a target; we can solve for the unknown to obtain a policy function for that component. This
essentially removes the need to solve it out in the greater model.

WARNING: This is still experimental and is likely not workable in the state space...
"""
struct SolvedBlock{BT<:AbstractBlock} <: AbstractBlock
    child::BT
    unknowns::OrderedSet{Symbol}
    targets::OrderedSet{Symbol}
    name::String
end

inputs(block::SolvedBlock) = inputs(block.child)
outputs(block::SolvedBlock) = collect(union(outputs(block.child), block.unknowns))

function solved(block::SimpleBlock, unknowns, targets)
    return SolvedBlock(block, unknowns, targets, "$(block.name) (solved)")
end

"""
    ComposedBlock{PT,CT}

A recursively defined block usually constructed by way of a topological sort. Each composed
block contains a parent block and a child block (the child is either a recursive block or a
root node).

This improvement to the combined block that behaves sort of like a linked list, but with
parametric types to enforce stability through a recursive evaluation.

To compose blocks, you should never have to call the constructor on its own, instead useres
are directed to either the `compose` or `model` functions.
"""
struct ComposedBlock{PT,CT} <: AbstractBlock
    parent::PT
    child::CT
    inputs::OrderedSet{Symbol}
    outputs::OrderedSet{Symbol}
    name::String
end

function compose(parent::AbstractBlock, child::AbstractBlock, name::String="block")
    invars = union(inputs(parent), inputs(child))
    outvars = OrderedSet{Symbol}(union(outputs(parent), outputs(child)))
    return ComposedBlock(
        parent, child, OrderedSet{Symbol}(setdiff(invars, outvars)), outvars, name
    )
end

inputs(block::ComposedBlock) = collect(block.inputs)
outputs(block::ComposedBlock) = collect(block.outputs)

# so far this is only used within the steady state residual computation
function (block::ComposedBlock)(x)
    y = merge(x, block.child(x))
    return merge(y, block.parent(y))
end

function construct_graph(blocks)
    dag = SimpleDiGraph(length(blocks))
    for n in 1:length(blocks)
        for output in outputs(blocks[n])
            m = findall(x -> in(output, x), inputs.(blocks))
            add_edge!.(Ref(dag), n, m)
        end
    end
    return dag
end

# while not type stable, it will almost never be a bottleneck
function model(blocks...; name::String="block")
    dag = construct_graph(blocks)
    return mapreduce(i -> blocks[i], (x, y) -> compose(y, x), topological_sort(dag))
end

include("jacobians.jl")
include("steady_state.jl")
# include("systems.jl")
include("misc.jl")
include("solutions.jl")
# include("state_space.jl")

end # module DynamicMacroeconomics
