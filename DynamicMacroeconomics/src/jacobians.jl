using ADTypes: KnownJacobianSparsityDetector

function DifferentiationInterface.jacobian(block::AbstractBlock, ss, inputs; kwargs...)
    return jacobian(block, ss, inputs, outputs(block); kwargs...)
end

function make_jacobian_arguments(block::SimpleBlock, ss::NamedTuple, unknowns)
    C = ss[setdiff(inputs(block), unknowns)]
    X = NamedTuple{unknowns}([fill(i, 3) for i in ss[unknowns]])
    return X, C
end

function make_jacobian_arguments(block::SimpleBlock, ss::ComponentVector, unknowns)
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
    return getdata(
        ComponentArray(block.sparsity, Axis(rowmap), Axis(colmap))[targets, unknowns]
    )
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

    return DifferentiationInterface.jacobian(
        (x, c) -> block(x, c)[targets], backend, X, Constant(C)
    )
end

# TODO: finish the combined block jacobian
function DifferentiationInterface.jacobian(
    blocks::CombinedBlock{F,NU,NO}, ss, unknowns, targets; backend=AutoForwardDiff()
) where {F,NU,NO}
    internal = tuple(intersect(union(inputs.(blocks)...), outputs(blocks))...)
    all_inputs = tuple(union(unknowns, internal)...)
    all_outputs = tuple(union(targets, internal)...)

    error("not yet implemented")
end
