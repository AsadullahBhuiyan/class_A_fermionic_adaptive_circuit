using LinearAlgebra

const TOUCHING_DEFAULT_PCUT = 0.75
const TOUCHING_DEFAULT_OMEGA_CUT = 1.5
const TOUCHING_DEFAULT_CUTOFF_POWER = 8

@inline function smooth_cutoff(x::Real, cutoff::Real, power::Int=TOUCHING_DEFAULT_CUTOFF_POWER)
    cutoff <= 0 && return 0.0
    return exp(-abs(x / cutoff)^power)
end

@inline function invert_2x2(a::ComplexF64, b::ComplexF64, d::ComplexF64)
    detinv = inv(a * d - b * b)
    return d * detinv, -b * detinv, -b * detinv, a * detinv
end

@inline function upper_touching_local_xbasis_entries(
    ω::Float64,
    k::Float64,
    Γ::Float64,
    η::Float64;
    pcut::Float64=TOUCHING_DEFAULT_PCUT,
    ωcut::Float64=TOUCHING_DEFAULT_OMEGA_CUT,
    cutoff_power::Int=TOUCHING_DEFAULT_CUTOFF_POWER,
)
    p = π / 2 - k
    if p <= 0
        return 0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im
    end

    z = ComplexF64(ω, η)
    s = 1 - p^2 / 2
    m = 1 - p
    Σ = (z^2 + 2p - 1 - sqrt(z^2 - 1) * sqrt(z^2 - (5 - 4p))) / (2 * (z - s))
    shift = im * Γ / 2

    a = z - s + shift
    b = ComplexF64(-m, 0.0)
    d = z + s - Σ + shift
    gpp, gpm, gmp, gmm = invert_2x2(a, b, d)

    weight = smooth_cutoff(p, pcut, cutoff_power) * smooth_cutoff(ω - 1, ωcut, cutoff_power)
    return weight * gpp, weight * gpm, weight * gmp, weight * gmm
end

@inline function lower_touching_local_xbasis_entries(
    ω::Float64,
    k::Float64,
    Γ::Float64,
    η::Float64;
    pcut::Float64=TOUCHING_DEFAULT_PCUT,
    ωcut::Float64=TOUCHING_DEFAULT_OMEGA_CUT,
    cutoff_power::Int=TOUCHING_DEFAULT_CUTOFF_POWER,
)
    p = k + π / 2
    if p <= 0
        return 0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im
    end

    z = ComplexF64(ω, η)
    s = -1 + p^2 / 2
    m = 1 - p
    Σ = (z^2 + 2p - 1 - sqrt(z^2 - 1) * sqrt(z^2 - (5 - 4p))) / (2 * (z - s))
    shift = im * Γ / 2

    a = z - s + shift
    b = ComplexF64(-m, 0.0)
    d = z + s - Σ + shift
    gpp, gpm, gmp, gmm = invert_2x2(a, b, d)

    weight = smooth_cutoff(p, pcut, cutoff_power) * smooth_cutoff(ω + 1, ωcut, cutoff_power)
    return weight * gpp, weight * gpm, weight * gmp, weight * gmm
end

@inline function local_touching_xbasis_entries(
    ω::Float64,
    k::Float64,
    Γ::Float64,
    η::Float64;
    pcut::Float64=TOUCHING_DEFAULT_PCUT,
    ωcut::Float64=TOUCHING_DEFAULT_OMEGA_CUT,
    cutoff_power::Int=TOUCHING_DEFAULT_CUTOFF_POWER,
)
    u11, u12, u21, u22 = upper_touching_local_xbasis_entries(
        ω,
        k,
        Γ,
        η;
        pcut=pcut,
        ωcut=ωcut,
        cutoff_power=cutoff_power,
    )
    l11, l12, l21, l22 = lower_touching_local_xbasis_entries(
        ω,
        k,
        Γ,
        η;
        pcut=pcut,
        ωcut=ωcut,
        cutoff_power=cutoff_power,
    )
    return u11 + l11, u12 + l12, u21 + l21, u22 + l22
end

function build_local_touching_pair!(
    data1::AbstractArray{ComplexF64,4},
    data2::AbstractArray{ComplexF64,4},
    ωs::AbstractVector,
    ks::AbstractVector,
    active_indices::AbstractVector;
    Γ::Float64,
    η::Float64,
    pcut::Float64=TOUCHING_DEFAULT_PCUT,
    ωcut::Float64=TOUCHING_DEFAULT_OMEGA_CUT,
    cutoff_power::Int=TOUCHING_DEFAULT_CUTOFF_POWER,
)
    @inbounds for j in eachindex(ks)
        k = ks[j]
        for i in active_indices
            g11, g12, g21, g22 = local_touching_xbasis_entries(
                ωs[i],
                k,
                Γ,
                η;
                pcut=pcut,
                ωcut=ωcut,
                cutoff_power=cutoff_power,
            )
            data1[i, j, 1, 1] = g11
            data1[i, j, 1, 2] = g12
            data1[i, j, 2, 1] = g21
            data1[i, j, 2, 2] = g22
            data2[i, j, 1, 1] = g11
            data2[i, j, 1, 2] = g12
            data2[i, j, 2, 1] = g21
            data2[i, j, 2, 2] = g22
        end
    end
    return nothing
end
