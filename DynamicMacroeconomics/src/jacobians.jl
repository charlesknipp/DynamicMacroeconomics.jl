export SparseJacobian, AbstractJacobian

## ABSTRACT JACOBIANS ######################################################################

abstract type AbstractJacobian end

# TODO: abhorrent but works...
struct SparseJacobian{NU,NO} <: AbstractJacobian
    dict::Dict
    inputs::NTuple{NU,Symbol}
    outputs::NTuple{NO,Symbol}
end

inputs(J::SparseJacobian) = J.inputs
outputs(J::SparseJacobian) = J.outputs

function SparseJacobian(M::AbstractMatrix, unknowns, targets)
    # visually disgusting, but is type stable
    jac_dict = Dict(
        target => Dict(
            unknown => centered(
                M[o, (3 * i - 2) : (3 * i)]
            ) for (i, unknown) in enumerate(unknowns)
        ) for (o, target) in enumerate(targets)
    )
    return SparseJacobian(jac_dict, unknowns, targets)
end

# identity Jacobian for each contemporaneous input
function SparseJacobian(::Type{T}, unknowns::NTuple{N,Symbol}) where {N,T<:Real}
    return SparseJacobian(
        Dict(i => Dict(i => centered(T[0, 1, 0])) for i in unknowns), unknowns, unknowns
    )
end

function Base.merge(J1::SparseJacobian, J2::SparseJacobian)
    inputs(J1) != inputs(J2) && @error("cannot combine jacobians with different inputs")
    new_dict = merge(J1.dict, J2.dict)
    return SparseJacobian(new_dict, inputs(J1), tuple(keys(new_dict)...))
end

Base.getindex(J::SparseJacobian, output::Symbol) = J.dict[output]
Base.getindex(J::SparseJacobian, output::Symbol, input::Symbol) = J.dict[output][input]

function subset(J::SparseJacobian, outputs)
    subdict = filter(p -> p.first in outputs, J.dict)
    return SparseJacobian(subdict, inputs(J), tuple(outputs...))
end

# ensures that lagged intermediates are not already functions of lagged variables
function resize_warn(∂::OffsetVector)
    ∂[-2] != 0 && @error("model has more than 1 lag")
    ∂[2] != 0 && @error("model has more than 1 lead")
    return OffsetArrays.centered(∂[-1:1])
end

# get biggest lead and lag
function offset_extrema(∂::OffsetVector)
    offsets = keys(∂)
    return first(offsets), last(offsets)
end

# get combined biggest lead and lag
function offset_extrema(∂Y::OffsetVector, ∂X::OffsetVector)
    return offset_extrema(∂Y) .+ offset_extrema(∂X)
end

# multiply two offset vectors to have closed form system multiplication
function chain_rule(∂Y∂X::OffsetVector, ∂X∂Z::OffsetVector)
    lag, lead = offset_extrema(∂Y∂X, ∂X∂Z)
    T = Base.promote_eltype(∂Y∂X, ∂X∂Z)
    ∂Y∂Z = OffsetVector(zeros(T, lead - lag + 1), lag - 1)
    for i in keys(∂Y∂X), j in keys(∂X∂Z)
        ∂Y∂Z[j+i] += ∂Y∂X[i] * ∂X∂Z[j]
    end
    return resize_warn(∂Y∂Z)
end

# type generic, which may be a problem for AD
function chain_rule(J_om::SparseJacobian, J_mi::SparseJacobian)
    outvars = outputs(J_om)
    m_list = union(inputs(J_om), outputs(J_mi))
    invars = inputs(J_mi)

    J_oi = Dict(o => Dict() for o in outvars)
    for o in outvars, i in invars
        Jout = nothing
        for m in m_list
            if (m in keys(J_om[o])) && (i in keys(J_mi[m]))
                if isnothing(Jout)
                    Jout = chain_rule(J_om[o][m], J_mi[m][i])
                else
                    Jout += chain_rule(J_om[o][m], J_mi[m][i])
                end
            end
        end
        if !isnothing(Jout)
            J_oi[o][i] = Jout
        end
    end

    return SparseJacobian(J_oi, invars, outvars)
end

## DIFFERENTIATION INTERFACE ###############################################################

function DifferentiationInterface.jacobian(block::AbstractBlock, ss, inputs; kwargs...)
    return jacobian(block, ss, inputs, outputs(block); kwargs...)
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

sparse_jacobian(block::SimpleBlock, unknowns) = sparse_jacobian(
    block, unknowns, block.outputs
)

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
    return SparseJacobian(M, unknowns, targets)
end

function DifferentiationInterface.jacobian(
    blocks::CombinedBlock{F,NU,NO}, ss, unknowns, targets; backend=AutoForwardDiff()
) where {F,NU,NO}
    all_outputs = outputs(blocks)
    total_jacobian = SparseJacobian(Float64, unknowns)
    for block in blocks
        intermediates = tuple((unknowns ∩ inputs(block)) ∪ (all_outputs ∩ inputs(block))...)
        block_jacobian = jacobian(block, ss, intermediates, outputs(block); backend)
        total_jacobian = merge(total_jacobian, chain_rule(block_jacobian, total_jacobian))
    end
    return subset(total_jacobian, targets)
end
