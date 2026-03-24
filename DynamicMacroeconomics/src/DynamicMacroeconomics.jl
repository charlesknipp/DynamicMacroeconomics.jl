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

export model, @simple, lead, lag, jacobian
export SimpleBlock, CombinedBlock
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

struct SimpleBlock{FT,NU,NO,SP} <: AbstractBlock
    functor::FT
    inputs::NTuple{NU,Symbol}
    outputs::NTuple{NO,Symbol}
    name::String
    sparsity::SP
end

inputs(block::SimpleBlock) = block.inputs
outputs(block::SimpleBlock) = block.outputs

macro simple(args...)
    func_def = MacroTools.splitdef(args[end])
    block_name = func_def[:name]
    inputs = tuple(func_def[:args]...)

    # build an anonymous function call with gensym("functor")
    functor = gensym(func_def[:name])
    func_def[:name] = functor
    outputs = build_expression!(func_def)

    # store the block sparsity for efficient evaluation of the Jacobian
    sparse_io = sparsity_detector(func_def[:body], inputs, outputs)
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
        if @capture(ex, return target_)
            if target isa Expr
                push!(targets, target.args...)
                return ex
            else
                # return singular targets as a tuple
                push!(targets, target)
                return :(return ($target,))
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

    return tuple(targets...)
end

function sparsity_detector(body, inputs, outputs)
    # make relevant substitutions for proper index handling
    exprmap = Dict{Symbol,Expr}()
    MacroTools.postwalk(body) do ex
        if @capture(ex, LHS_ = RHS_)
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

function (block::SimpleBlock)(x::NamedTuple)
    return NamedTuple{block.outputs}(block.functor(x[block.inputs]...))
end

function (block::SimpleBlock)(x::NamedTuple, c::NamedTuple)
    return block(merge(c, x))
end

function (block::SimpleBlock)(var::Symbol, x::AbstractVector{<:Real}, c::NamedTuple)
    return block(NamedTuple{(var,)}((x,)), c)
end

# workaround for AD using OffsetArrays
offset_component(X::Real, offset_range) = X
offset_component(X::AbstractVector, offset_range) = OffsetVector(X, offset_range)

function (block::SimpleBlock)(x::ComponentArray)
    fargs = [offset_component(x[i], block.sparsity[i]) for i in inputs(block)]
    return ComponentVector(collect(block.functor(fargs...)), Axis(block.outputs))
end

# when taking partials given a constant (not type stable)
(block::SimpleBlock)(x::ComponentArray, c::ComponentArray) = block([x; c])

struct CombinedBlock{BS<:Tuple,NU,NO} <: AbstractBlock
    dag::SimpleDiGraph
    blocks::BS
    inputs::NTuple{NU,Symbol}
    outputs::NTuple{NO,Symbol}
    name::String
end

Base.length(model::CombinedBlock) = length(model.blocks)
Base.@propagate_inbounds Base.getindex(model::CombinedBlock, i) = model.blocks[i]

Base.iterate(model::CombinedBlock) = iterate(model.blocks)
Base.iterate(model::CombinedBlock, state) = iterate(model.blocks, state)

function combined!(blocks::CombinedBlock, out, x)
    for block in blocks
        out = merge(out, block(x))
        x = merge(x, out)
    end
    return out
end

function (blocks::CombinedBlock)(x::NamedTuple)
    out = (;)
    return NamedTuple{outputs(blocks)}(combined!(block, out, x))
end

function (blocks::CombinedBlock)(x::ComponentVector{T}) where {T}
    out = ComponentVector{T}()
    return combined!(blocks, out, x)
end

inputs(block::CombinedBlock) = block.inputs
outputs(block::CombinedBlock) = block.outputs

# take blocks and construct a digraph
function model(blocks...; name::String="block")
    dag = SimpleDiGraph(length(blocks))
    for (n, block) in enumerate(blocks)
        for output in outputs(block)
            m = findall(x -> in(output, x), inputs.(blocks))
            add_edge!.(Ref(dag), n, m)
        end
    end

    blocks = blocks[topological_sort(dag)]
    invars, outvars = union(inputs.(blocks)...), union(outputs.(blocks)...)
    return CombinedBlock(
        dag, blocks, tuple(setdiff(invars, outvars)...), tuple(outvars...), name
    )
end

include("jacobians.jl")
include("steady_state.jl")
include("systems.jl")
include("misc.jl")
include("solutions.jl")
# include("state_space.jl")

end # module DynamicMacroeconomics
