module DynamicMacroeconomics

using Distributions
using Random

using GeneralisedFilters
using SSMProblems
using MatrixEquations
using LinearAlgebra
using ToeplitzMatrices

using DifferentiationInterface
import ForwardDiff

using NonlinearSolve

using Reexport
using MacroTools
using Graphs

@reexport using SSMProblems, GeneralisedFilters

export model, @simple, lead, lag

# for steady state evaluations
lead(x::Real) = x
lag(x::Real) = x
contemp(x::Real) = x

# for partial derivatives
lead(x::AbstractVector{<:Real}) = x[3]
lag(x::AbstractVector{<:Real}) = x[1]
contemp(x::AbstractVector{<:Real}) = x[2]

abstract type AbstractBlock end

function Base.show(io::IO, b::AbstractBlock)
    inputs  = join(b.inputs, ", ")
    outputs = join(b.outputs, ", ")
    print(io, "$(b.name): ($inputs) → ($outputs)")
end

function inputs(::AbstractBlock) end
function outputs(::AbstractBlock) end

struct SimpleBlock{FT,NU,NO} <: AbstractBlock
    functor::FT
    inputs::NTuple{NU,Symbol}
    outputs::NTuple{NO,Symbol}
    name::String
end

inputs(block::SimpleBlock) = block.inputs
outputs(block::SimpleBlock) = block.outputs

macro simple(args...)
    # split the function definition
    func_def = MacroTools.splitdef(args[end])

    # extract the name to define the block
    block_name = func_def[:name]
    inputs = tuple(func_def[:args]...)

    # build an anonymous function call with gensym("functor")
    functor = gensym(func_def[:name])
    func_def[:name] = functor
    outputs = build_expression!(func_def)

    # return the quoted expression
    return MacroTools.@q begin
        $(MacroTools.combinedef(func_def))
        $(esc(block_name)) = $(SimpleBlock)(
            $(functor), $(inputs), $(outputs), $(string(block_name))
        )
    end
end

function build_expression!(func_dict)
    # capture the targets in the return statement
    targets = Symbol[]
    MacroTools.postwalk(func_dict[:body]) do ex
        if @capture(ex, return target_)
            if target isa Expr
                push!(targets, target.args...)
            else
                push!(targets, target)
            end
        else
            return ex
        end
    end

    # encase time series with comtemp()
    new_ex = MacroTools.postwalk(func_dict[:body]) do ex
        if ex isa Symbol && ex in func_dict[:args]
            return :(contemp($ex))
        else
            return ex
        end
    end

    # undo extraneous contemp calls for lags/leads
    fixed_ex = MacroTools.postwalk(new_ex) do ex
        if @capture(ex, f_(contemp(var_))) && f in [:lead, :lag]
            return :($f($var))
        else
            return ex
        end
    end

    func_dict[:body] = fixed_ex
    return tuple(targets...)
end

# makes blocks callable
function (block::SimpleBlock)(x)
    out = block.functor(x...)
    return NamedTuple{outputs(block)}(out)
end

# TODO: still needs work, but is at least type stable
function get_partials(block::SimpleBlock, ss::NamedTuple, backend=AutoForwardDiff())
    invars = inputs(block)
    ssvars = [ss[invars]...]
    stacked_ss = repeat(ssvars, 1, 3)

    # not too happy about this component
    ∂s = map(outputs(block)) do out
        ∇x = DifferentiationInterface.gradient(
            x -> block(eachrow(x))[out], backend, stacked_ss
        )

        Dict(zip(invars, eachrow(∇x)))
    end
    return Dict(zip(outputs(block), ∂s))
end

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

function (blocks::CombinedBlock)(x)
    out = (;)
    for block in blocks
        invars = inputs(block)
        out = merge(out, block(x[invars]))
        x = merge(x, out)
    end
    return NamedTuple{outputs(blocks)}(out)
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
    return CombinedBlock(dag, blocks, tuple(setdiff(invars, outvars)...), tuple(outvars...), name)
end

function get_partials(::CombinedBlock, ::NamedTuple, backend)
    throw("not yet implemented")
end

include("steady_state.jl")
include("systems.jl")
include("misc.jl")
include("solutions.jl")
include("state_space.jl")

end # module DynamicMacroeconomics
