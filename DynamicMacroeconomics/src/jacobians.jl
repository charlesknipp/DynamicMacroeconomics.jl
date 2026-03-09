export ToeplitzSymbol, BlockJacobian, AbstractJacobian

import Base: *, +

## SPARSE IMPULSE ##########################################################################

# TODO: get rid of the offsets dict and move to a AbstractToeplitz type definition
struct ToeplitzSymbol{Tv,Ti} <: AbstractSparseVector{Tv,Ti}
    offsets::Dict{Ti,Ti}
    values::Vector{Tv}
end

ToeplitzSymbol(::Type{T}) where {T} = ToeplitzSymbol(Dict{Int64,Int64}(), T[])

# TODO: this is definitely slow
function ToeplitzSymbol(A::SparseVector{T,Int64}, offset) where {T}
    B = ToeplitzSymbol(T)
    for i in SparseArrays.nonzeroinds(A)
        B[i + offset] = A[i]
    end
    return B
end

function ToeplitzSymbol(A::OffsetVector{T,SparseVector{T,Int64}}) where {T}
    return ToeplitzSymbol(A.parent, A.offsets[1])
end

SparseArrays.nonzeroinds(A::ToeplitzSymbol) = collect(keys(A.offsets))
SparseArrays.nonzeros(A::ToeplitzSymbol) = getfield(A, :values)
SparseArrays.rowvals(A::ToeplitzSymbol) = SparseArrays.nonzeroinds(A)

function Base.size(A::ToeplitzSymbol)
    all_keys = keys(A.offsets)
    return (maximum(all_keys; init=0) - minimum(all_keys; init=1) + 1,)
end

function Base.getindex(A::ToeplitzSymbol{T}, i::Integer) where {T}
    if i in keys(A.offsets)
        return A.values[A.offsets[i]]
    else
        return zero(T)
    end
end

# TODO: return a ToeplitzSymbol instead of a regular vector...
function Base.getindex(A::ToeplitzSymbol{T}, range::AbstractUnitRange) where {T}
    return T[A[i] for i in range]
end

function Base.setindex!(A::ToeplitzSymbol, value, i::Integer)
    if i in keys(A.offsets)
        setindex!(A.values, value, A.offsets[i])
    else
        A.offsets[i] = length(A.values) + 1
        push!(A.values, value)
    end
end

function (*)(A::ToeplitzSymbol, B::ToeplitzSymbol)
    C = ToeplitzSymbol(Base.promote_eltypeof(A.values, B.values))
    for i in keys(A.offsets), j in keys(B.offsets)
        C[i + j] += A[i] * B[j]
    end
    return C
end

function (+)(A::ToeplitzSymbol, B::ToeplitzSymbol)
    C = ToeplitzSymbol(Base.promote_eltypeof(A.values, B.values))
    for i in union(keys(A.offsets), keys(B.offsets))
        C[i] = A[i] + B[i]
    end
    return C
end

# for use in sequence jacobian solver
function ToeplitzMatrices.Toeplitz(A::ToeplitzSymbol{AT}, T::Integer) where {AT}
    spc = spzeros(AT, T)
    spr = spzeros(AT, T)
    for i in keys(A.offsets)
        i >= 0 && (spc[1 + i] = A[i])
        i <= 0 && (spr[1 - i] = A[i])
    end
    return Toeplitz(spr, spc)
end

## ABSTRACT JACOBIANS ######################################################################

abstract type AbstractJacobian{T} end

# TODO: replace the matrix with a vector and use CSR sparsity for indexing
struct BlockJacobian{T} <: AbstractJacobian{T}
    partials::Matrix{ToeplitzSymbol{T,Int64}}
    inputs::Dict{Symbol,Int64}
    outputs::Dict{Symbol,Int64}
end

DynamicMacroeconomics.inputs(A::BlockJacobian) = keys(A.inputs)
DynamicMacroeconomics.outputs(A::BlockJacobian) = keys(A.outputs)

function Base.show(io::IO, A::BlockJacobian{T}) where {T}
    inputs = join(keys(A.inputs), ", ")
    outputs = join(keys(A.outputs), ", ")
    return print(io, "BlockJacobian{$T}:\n  ($inputs) → ($outputs)")
end

function BlockJacobian(::Type{T}, inputs, outputs) where {T}
    base_impulse = ToeplitzSymbol(T)
    return BlockJacobian(
        fill(deepcopy(base_impulse), length(outputs), length(inputs)),
        Dict(var => i for (i, var) in enumerate(inputs)),
        Dict(var => i for (i, var) in enumerate(outputs)),
    )
end

function BlockJacobian(::Type{T}, varnames) where {T}
    id = ToeplitzSymbol(T)
    id[0] = 1
    A = BlockJacobian(T, varnames, varnames)
    for var in varnames
        A[var, var] = id
    end
    return A
end

function BlockJacobian(M::AbstractMatrix{T}, unknowns, targets) where {T}
    A = BlockJacobian(T, unknowns, targets)
    for (i, unknown) in enumerate(unknowns), (o, target) in enumerate(targets)
        A[target, unknown] = ToeplitzSymbol(centered(M[o, (3 * i - 2):(3 * i)]))
    end
    return A
end

function Base.getindex(A::BlockJacobian, output::Symbol, input::Symbol)
    return getindex(A.partials, A.outputs[output], A.inputs[input])
end

function Base.setindex!(A::BlockJacobian, val, output::Symbol, input::Symbol)
    # this should ALMOST never be used outside of Base.*
    return setindex!(A.partials, val, A.outputs[output], A.inputs[input])
end

function (*)(A::BlockJacobian{AT}, B::BlockJacobian{BT}) where {AT,BT}
    invars = inputs(B)
    intermediates = union(inputs(A), outputs(B))
    outvars = outputs(A)
    C = BlockJacobian(Base.promote_type(AT, BT), invars, outvars)
    for o in outvars, i in invars, m in intermediates
        if m in inputs(A)
            C[o, i] += A[o, m] * B[m, i]
        end
    end
    return C
end

function Base.merge(A::BlockJacobian{AT}, B::BlockJacobian{BT}) where {AT,BT}
    invars = intersect(inputs(A), inputs(B))
    outvars = symdiff(outputs(A), outputs(B))
    C = BlockJacobian(Base.promote_type(AT, BT), invars, outvars)
    for J in (A, B), o in outputs(J), i in inputs(J)
        C[o, i] = J[o, i]
    end
    return C
end

# TODO: please get rid of this extraneous allocation
function subset(A::BlockJacobian{T}, targets) where {T}
    B = BlockJacobian(T, inputs(A), targets)
    for o in targets, i in inputs(A)
        B[o, i] = A[o, i]
    end
    return B
end

## DIFFERENTIATION INTERFACE ###############################################################

function DifferentiationInterface.jacobian(block::AbstractBlock, ss, unknowns; kwargs...)
    return jacobian(block, ss, unknowns, outputs(block); kwargs...)
end

function make_jacobian_arguments(block::AbstractBlock, ss::NamedTuple, unknowns)
    C = ss[setdiff(inputs(block), unknowns)]
    X = NamedTuple{unknowns}([fill(i, 3) for i in ss[unknowns]])
    return X, C
end

function make_jacobian_arguments(block::AbstractBlock, ss::ComponentVector, unknowns)
    X, C = make_jacobian_arguments(block, (; ss...), unknowns)
    return ComponentVector(; X...), ComponentVector(; C...)
end

function sparse_jacobian(block::SimpleBlock, unknowns)
    return sparse_jacobian(block, unknowns, block.outputs)
end

function sparse_jacobian(block::SimpleBlock, unknowns, targets)
    rowmap = DynamicMacroeconomics.named_tuple(block.outputs)
    colmap = map(DynamicMacroeconomics.named_tuple(block.inputs)) do j
        (3 * (j - 1) + 1):(3 * j)
    end
    sparsity_pattern = getdata(
        ComponentArray(block.sparsity, Axis(rowmap), Axis(colmap))[targets, unknowns]
    )
    return KnownJacobianSparsityDetector(sparsity_pattern)
end

function DifferentiationInterface.jacobian(
    block::SimpleBlock, ss, unknowns, targets; backend=AutoForwardDiff(), sparse=true
)
    targets = tuple(intersect(targets, outputs(block))...)
    X, C = make_jacobian_arguments(block, ss, unknowns)
    if sparse
        backend = AutoSparse(
            backend, sparse_jacobian(block, unknowns, targets), GreedyColoringAlgorithm()
        )
    end
    M = DifferentiationInterface.jacobian(
        (x, c) -> block(x, c)[targets], backend, X, Constant(C)
    )
    return BlockJacobian(M, unknowns, targets)
end

function DifferentiationInterface.jacobian(
    blocks::CombinedBlock{F,NU,NO}, ss, unknowns, targets; backend=AutoForwardDiff()
) where {F,NU,NO}
    all_outputs = outputs(blocks)
    total_jacobian = BlockJacobian(eltype(ss), unknowns)
    for block in blocks
        intermediates = tuple((unknowns ∩ inputs(block)) ∪ (all_outputs ∩ inputs(block))...)
        block_jacobian = jacobian(block, ss, intermediates, outputs(block); backend)
        total_jacobian = merge(total_jacobian, block_jacobian * total_jacobian)
    end
    return subset(total_jacobian, targets)
end
