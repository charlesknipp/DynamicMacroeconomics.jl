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

# only simple blocks are currently defined
struct SimpleBlock{FT,NU,NO,MT} <: AbstractBlock
    functor::FT
    inputs::NTuple{NU,Symbol}
    outputs::NTuple{NO,Symbol}
    metadata::MT
end

inputs(block::SimpleBlock) = block.inputs
outputs(block::SimpleBlock) = block.outputs

function show_variables(vars::NTuple{N,Symbol}) where {N}
    join(vars, ", ")
end

function Base.show(io::IO, b::SimpleBlock)
    block_name = split(string(typeof(b.functor).name.name), "#"; keepempty=false)[2]
    inputs = show_variables(b.inputs)
    outputs = show_variables(b.outputs)
    print(io, "$block_name: ($inputs) → ($outputs)")
end

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

    # grab the offsets and store that in metadata
    metadata = setdiff(func_def[:args], inputs)

    # return the quoted expression
    return MacroTools.@q begin
        $(MacroTools.combinedef(func_def))
        $(esc(block_name)) = $(SimpleBlock)($(functor), $(inputs), $(outputs), $(metadata))
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
function (block::SimpleBlock)(x...)
    out = block.functor(x...)
    return NamedTuple{outputs(block)}(out)
end

# TODO: this needs some work...
function get_partials(block::SimpleBlock, ss::NamedTuple, backend=AutoForwardDiff())
    invars = inputs(block)
    ssvars = [ss[invars]...]
    stacked_ss = repeat(ssvars, 1, 3)

    ∂s = Dict()
    for out in outputs(block)
        ∇x = DifferentiationInterface.gradient(
            x -> block(collect(eachrow(x))...)[out], backend, stacked_ss
        )

        ∂s[out] = Dict(zip(invars, eachrow(∇x)))
    end
    return ∂s
end

struct GraphicalModel{N,NU,NT}
    dag::SimpleDiGraph
    blocks::NTuple{N,AbstractBlock}
    order::Vector{Int64}

    unknowns::NTuple{NU,Symbol}
    targets::NTuple{NT,Symbol}
end

# TODO: improve pretty printing
function Base.show(io::IO, model::GraphicalModel)
    println(io, join(model.blocks, "\n"))
end

Base.length(::GraphicalModel{N}) where {N} = N

Base.@propagate_inbounds Base.getindex(model::GraphicalModel, i) = model.blocks[model.order[i]]

# take blocks and construct a digraph
function model(blocks...)
    pool = collect(blocks)
    graph = SimpleDiGraph(length(pool))
    for (n, block) in enumerate(pool)
        for output in outputs(block)
            m = findall(x -> in(output, x), inputs.(pool))
            add_edge!.(Ref(graph), n, m)
        end
    end
    order = topological_sort(graph)

    invars = union((inputs(pool[i]) for i in order)...)
    outvars = union((outputs(pool[i]) for i in order)...)

    unknowns = tuple(setdiff(invars, outvars)...)
    targets = tuple(setdiff(outvars, invars)...)

    return GraphicalModel(graph, blocks, order, unknowns, targets)
end

include("steady_state.jl")
include("systems.jl")
include("misc.jl")
include("solutions.jl")
include("state_space.jl")

end # module DynamicMacroeconomics
