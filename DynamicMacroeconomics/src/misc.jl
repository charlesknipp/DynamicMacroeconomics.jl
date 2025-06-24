export StructuralMvNormal

## STRUCTURAL NOISE MVNORMAL ###############################################################

"""
    StructuralMvNormal(μ, R)

For multivariate Gaussian distributions with iid noise which is treated more like a control.
This becomes especially useful when defining macroeconomic models with structural noise,
hence the naming convention. The following VAR
```
x[t] = A * x[t-1] + B * ε[t]    s.t.    ε[t] ~ MvNormal(I)
```
then becomes
```
StructuralMvNormal(A * x[t-1], B)
```
which interfaces with `Distributions` and thus `Turing`.
"""
struct StructuralMvNormal{T<:Real,RT<:AbstractMatrix,MT<:AbstractVector} <: AbstractMvNormal
    μ::MT
    R::RT
end

function StructuralMvNormal(μ::AbstractVector{T}, R::AbstractMatrix{T}) where {T<:Real}
    size(R, 1) == length(μ) || throw(DimensionMismatch("The dimensions of μ and Σ are inconsistent."))
    return StructuralMvNormal{T,typeof(R),typeof(μ)}(μ, R)
end

function StructuralMvNormal(μ::AbstractVector{<:Real}, R::AbstractMatrix{<:Real})
    T = Base.promote_eltype(μ, R)
    StructuralMvNormal(convert(AbstractArray{T}, μ), convert(AbstractArray{T}, R))
end

function StructuralMvNormal(μ::AbstractVector{<:Real}, R::AbstractVector{<:Real})
    return StructuralMvNormal(μ, reshape(R, length(R), 1))
end

Base.eltype(::Type{<:StructuralMvNormal{T}}) where {T} = T
Base.length(d::StructuralMvNormal) = length(d.μ)

function generate_noise(rng::AbstractRNG, d::AbstractMvNormal, x::AbstractMatrix{<:Real})
    return rand(rng, d, size(x, 2))
end

function generate_noise(rng::AbstractRNG, d::AbstractMvNormal, ::AbstractVector{<:Real})
    return rand(rng, d)
end

function Distributions._rand!(rng::AbstractRNG, d::StructuralMvNormal, x::VecOrMat)
    noise_dist = MvNormal(I(size(d.R, 2)))
    mul!(x, d.R, generate_noise(rng, noise_dist, x))
    x .+= d.μ
    return x
end

function Distributions._logpdf(d::StructuralMvNormal, x::AbstractVector{<:Real})
    # NOTE: disregard the logdet since noise is iid (not yet confirmed)
    sqmahal = LinearAlgebra.norm_sqr(d.R \ (x - d.μ))
    return -(sqmahal + size(d.R, 2) * log(2pi)) / 2
end
