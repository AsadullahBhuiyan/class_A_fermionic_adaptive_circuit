using JLD2
using Printf

include("touching_model.jl")

const TOUCHING_WZW_DIR_PREFIX = "touching_wzw_cubic_scan"

function gamma_token(Γ::Real)
    return replace(@sprintf("%.3f", float(Γ)), "." => "p")
end

distribution_tag(::Val{:sign}) = "sign"
distribution_tag(::Val{:unity}) = "unity"
distribution_tag(::Val{:dyson_sign}) = "dyson_sign"
distribution_tag(::Val{:dyson_unity}) = "dyson_unity"

@inline function distribution_value(::Val{:sign}, ω::Float64)
    return ω > 0 ? 1.0 : (ω < 0 ? -1.0 : 0.0)
end

@inline distribution_value(::Val{:unity}, ω::Float64) = 1.0
@inline distribution_value(::Val{:dyson_sign}, ω::Float64) = distribution_value(Val(:sign), ω)
@inline distribution_value(::Val{:dyson_unity}, ω::Float64) = 1.0

@inline function touching_keldysh_entries(
    ::Val{:sign},
    ω::Float64,
    Γ::Float64,
    g11::ComplexF64,
    g12::ComplexF64,
    g21::ComplexF64,
    g22::ComplexF64,
    ga11::ComplexF64,
    ga12::ComplexF64,
    ga21::ComplexF64,
    ga22::ComplexF64,
)
    fω = distribution_value(Val(:sign), ω)
    return (
        fω * (g11 - ga11),
        fω * (g12 - ga12),
        fω * (g21 - ga21),
        fω * (g22 - ga22),
    )
end

@inline function touching_keldysh_entries(
    ::Val{:unity},
    ω::Float64,
    Γ::Float64,
    g11::ComplexF64,
    g12::ComplexF64,
    g21::ComplexF64,
    g22::ComplexF64,
    ga11::ComplexF64,
    ga12::ComplexF64,
    ga21::ComplexF64,
    ga22::ComplexF64,
)
    return (
        g11 - ga11,
        g12 - ga12,
        g21 - ga21,
        g22 - ga22,
    )
end

@inline function touching_keldysh_entries(
    mode::Union{Val{:dyson_sign},Val{:dyson_unity}},
    ω::Float64,
    Γ::Float64,
    g11::ComplexF64,
    g12::ComplexF64,
    g21::ComplexF64,
    g22::ComplexF64,
    ga11::ComplexF64,
    ga12::ComplexF64,
    ga21::ComplexF64,
    ga22::ComplexF64,
)
    fω = distribution_value(mode, ω)
    p11 = g11 * ga11 + g12 * ga21
    p12 = g11 * ga12 + g12 * ga22
    p21 = g21 * ga11 + g22 * ga21
    p22 = g21 * ga12 + g22 * ga22
    return (
        fω * ((g11 - ga11) + im * Γ * p11),
        fω * ((g12 - ga12) + im * Γ * p12),
        fω * ((g21 - ga21) + im * Γ * p21),
        fω * ((g22 - ga22) + im * Γ * p22),
    )
end

function touching_wzw_output_dir(Γs::AbstractVector, dδ::Float64, dp::Float64, δmax::Float64, pmax::Float64, distribution_mode::Symbol)
    gtag = join(gamma_token.(Γs), "_")
    return "$(TOUCHING_WZW_DIR_PREFIX)_$(String(distribution_mode))_dδ$(gamma_token(dδ))_dp$(gamma_token(dp))_δmax$(gamma_token(δmax))_pmax$(gamma_token(pmax))_Γ$(gtag)"
end

@inline function trace_prod_2x2(
    a11::ComplexF64,
    a12::ComplexF64,
    a21::ComplexF64,
    a22::ComplexF64,
    b11::ComplexF64,
    b12::ComplexF64,
    b21::ComplexF64,
    b22::ComplexF64,
    c11::ComplexF64,
    c12::ComplexF64,
    c21::ComplexF64,
    c22::ComplexF64,
)
    m11 = a11 * b11 + a12 * b21
    m12 = a11 * b12 + a12 * b22
    m21 = a21 * b11 + a22 * b21
    m22 = a21 * b12 + a22 * b22
    return m11 * c11 + m12 * c21 + m21 * c12 + m22 * c22
end

@inline function trace_product(
    dataA::Array{ComplexF64,3},
    iA::Int,
    jA::Int,
    dataB::Array{ComplexF64,3},
    iB::Int,
    jB::Int,
    dataC::Array{ComplexF64,3},
    iC::Int,
    jC::Int,
)
    return trace_prod_2x2(
        dataA[iA, jA, 1], dataA[iA, jA, 2], dataA[iA, jA, 3], dataA[iA, jA, 4],
        dataB[iB, jB, 1], dataB[iB, jB, 2], dataB[iB, jB, 3], dataB[iB, jB, 4],
        dataC[iC, jC, 1], dataC[iC, jC, 2], dataC[iC, jC, 3], dataC[iC, jC, 4],
    )
end

function build_touching_patch_components(
    δ_values::AbstractVector,
    p_values::AbstractVector;
    Γ::Float64,
    η::Float64,
    distribution_mode::Symbol,
    pcut::Float64,
    ωcut::Float64,
    cutoff_power::Int,
)
    mode = Val(distribution_mode)
    Nδ = length(δ_values)
    Np = length(p_values)

    upper_GR = Array{ComplexF64}(undef, Nδ, Np, 4)
    upper_GA = Array{ComplexF64}(undef, Nδ, Np, 4)
    upper_GK = Array{ComplexF64}(undef, Nδ, Np, 4)
    lower_GR = Array{ComplexF64}(undef, Nδ, Np, 4)
    lower_GA = Array{ComplexF64}(undef, Nδ, Np, 4)
    lower_GK = Array{ComplexF64}(undef, Nδ, Np, 4)

    @inbounds for jp in eachindex(p_values)
        p = p_values[jp]
        kup = π / 2 - p
        klo = -π / 2 + p
        for iδ in eachindex(δ_values)
            δ = δ_values[iδ]

            ωup = 1 + δ
            g11, g12, g21, g22 = upper_touching_local_xbasis_entries(
                ωup,
                kup,
                Γ,
                η;
                pcut=pcut,
                ωcut=ωcut,
                cutoff_power=cutoff_power,
            )
            ga11 = conj(g11)
            ga12 = conj(g12)
            ga21 = conj(g21)
            ga22 = conj(g22)
            upper_GR[iδ, jp, 1] = g11
            upper_GR[iδ, jp, 2] = g12
            upper_GR[iδ, jp, 3] = g21
            upper_GR[iδ, jp, 4] = g22
            upper_GA[iδ, jp, 1] = ga11
            upper_GA[iδ, jp, 2] = ga12
            upper_GA[iδ, jp, 3] = ga21
            upper_GA[iδ, jp, 4] = ga22
            k11, k12, k21, k22 = touching_keldysh_entries(mode, ωup, Γ, g11, g12, g21, g22, ga11, ga12, ga21, ga22)
            upper_GK[iδ, jp, 1] = k11
            upper_GK[iδ, jp, 2] = k12
            upper_GK[iδ, jp, 3] = k21
            upper_GK[iδ, jp, 4] = k22

            ωlo = -1 + δ
            g11, g12, g21, g22 = lower_touching_local_xbasis_entries(
                ωlo,
                klo,
                Γ,
                η;
                pcut=pcut,
                ωcut=ωcut,
                cutoff_power=cutoff_power,
            )
            ga11 = conj(g11)
            ga12 = conj(g12)
            ga21 = conj(g21)
            ga22 = conj(g22)
            lower_GR[iδ, jp, 1] = g11
            lower_GR[iδ, jp, 2] = g12
            lower_GR[iδ, jp, 3] = g21
            lower_GR[iδ, jp, 4] = g22
            lower_GA[iδ, jp, 1] = ga11
            lower_GA[iδ, jp, 2] = ga12
            lower_GA[iδ, jp, 3] = ga21
            lower_GA[iδ, jp, 4] = ga22
            k11, k12, k21, k22 = touching_keldysh_entries(mode, ωlo, Γ, g11, g12, g21, g22, ga11, ga12, ga21, ga22)
            lower_GK[iδ, jp, 1] = k11
            lower_GK[iδ, jp, 2] = k12
            lower_GK[iδ, jp, 3] = k21
            lower_GK[iδ, jp, 4] = k22
        end
    end

    return (
        upper_GR=upper_GR,
        upper_GA=upper_GA,
        upper_GK=upper_GK,
        lower_GR=lower_GR,
        lower_GA=lower_GA,
        lower_GK=lower_GK,
    )
end

function patch_triangle_value(
    GR::Array{ComplexF64,3},
    GA::Array{ComplexF64,3},
    GK::Array{ComplexF64,3},
    n1::Int,
    m1::Int,
    n2::Int,
    m2::Int;
    k_sign::Int,
    prefactor::Float64,
)
    Nδ = size(GR, 1)
    Np = size(GR, 2)
    acc = 0.0 + 0.0im

    @inbounds for jp in 1:Np
        jp1 = jp + k_sign * m1
        jp2 = jp + k_sign * (m1 + m2)
        if !(1 <= jp1 <= Np && 1 <= jp2 <= Np)
            continue
        end
        for iδ in 1:Nδ
            iδ1 = iδ - n1
            iδ2 = iδ - n1 - n2
            if !(1 <= iδ1 <= Nδ && 1 <= iδ2 <= Nδ)
                continue
            end

            acc += trace_product(GR, iδ, jp, GA, iδ1, jp1, GK, iδ2, jp2)
            acc += trace_product(GK, iδ, jp, GR, iδ1, jp1, GA, iδ2, jp2)
            acc += trace_product(GA, iδ, jp, GK, iδ1, jp1, GR, iδ2, jp2)
            acc -= trace_product(GK, iδ, jp, GK, iδ1, jp1, GK, iδ2, jp2)
        end
    end

    return prefactor * acc
end

function touching_triangle_value(components, n1::Int, m1::Int, n2::Int, m2::Int, prefactor::Float64)
    upper = patch_triangle_value(
        components.upper_GR,
        components.upper_GA,
        components.upper_GK,
        n1,
        m1,
        n2,
        m2;
        k_sign=+1,
        prefactor=prefactor,
    )
    lower = patch_triangle_value(
        components.lower_GR,
        components.lower_GA,
        components.lower_GK,
        n1,
        m1,
        n2,
        m2;
        k_sign=-1,
        prefactor=prefactor,
    )
    return upper + lower
end

function mixed_derivative(values::NTuple{4,ComplexF64}, hx::Float64, hy::Float64)
    fpp, fpm, fmp, fmm = values
    return (fpp - fpm - fmp + fmm) / (4 * hx * hy)
end

function compute_wzw_stencil(
    components;
    dδ::Float64,
    dp::Float64,
)
    f_Ω1k2_pp = touching_triangle_value(components, +1, 0, 0, +1, dδ * dp / (2π)^2)
    f_Ω1k2_pm = touching_triangle_value(components, +1, 0, 0, -1, dδ * dp / (2π)^2)
    f_Ω1k2_mp = touching_triangle_value(components, -1, 0, 0, +1, dδ * dp / (2π)^2)
    f_Ω1k2_mm = touching_triangle_value(components, -1, 0, 0, -1, dδ * dp / (2π)^2)

    f_k1Ω2_pp = touching_triangle_value(components, 0, +1, +1, 0, dδ * dp / (2π)^2)
    f_k1Ω2_pm = touching_triangle_value(components, 0, +1, -1, 0, dδ * dp / (2π)^2)
    f_k1Ω2_mp = touching_triangle_value(components, 0, -1, +1, 0, dδ * dp / (2π)^2)
    f_k1Ω2_mm = touching_triangle_value(components, 0, -1, -1, 0, dδ * dp / (2π)^2)

    dΩ1dk2 = mixed_derivative((f_Ω1k2_pp, f_Ω1k2_pm, f_Ω1k2_mp, f_Ω1k2_mm), dδ, dp)
    dk1dΩ2 = mixed_derivative((f_k1Ω2_pp, f_k1Ω2_pm, f_k1Ω2_mp, f_k1Ω2_mm), dp, dδ)

    return (
        f_Ω1k2_pp=f_Ω1k2_pp,
        f_Ω1k2_pm=f_Ω1k2_pm,
        f_Ω1k2_mp=f_Ω1k2_mp,
        f_Ω1k2_mm=f_Ω1k2_mm,
        f_k1Ω2_pp=f_k1Ω2_pp,
        f_k1Ω2_pm=f_k1Ω2_pm,
        f_k1Ω2_mp=f_k1Ω2_mp,
        f_k1Ω2_mm=f_k1Ω2_mm,
        dΩ1dk2=dΩ1dk2,
        dk1dΩ2=dk1dΩ2,
        anti_wzw=0.5 * (dΩ1dk2 - dk1dΩ2),
        sym_wzw=0.5 * (dΩ1dk2 + dk1dΩ2),
    )
end

function save_touching_wzw_metadata(output_dir::AbstractString, metadata::NamedTuple)
    jldsave(joinpath(output_dir, "metadata.jld2"); metadata...)
    return nothing
end

function save_touching_wzw_gamma(output_dir::AbstractString, Γ::Float64, result::NamedTuple)
    filepath = joinpath(output_dir, "gamma_$(gamma_token(Γ)).jld2")
    jldsave(filepath; Γ, result...)
    return filepath
end

function search_touching_wzw_cubic(;
    Γs::AbstractVector=[20.0, 40.0, 80.0],
    dδ::Float64=0.01,
    δmax::Float64=3.0,
    dp::Float64=0.001,
    pmax::Float64=1.2,
    η::Float64=1e-4,
    distribution_mode::Symbol=:dyson_sign,
    pcut::Float64=TOUCHING_DEFAULT_PCUT,
    ωcut::Float64=TOUCHING_DEFAULT_OMEGA_CUT,
    cutoff_power::Int=TOUCHING_DEFAULT_CUTOFF_POWER,
    output_dir::Union{Nothing, AbstractString}=nothing,
)
    @assert distribution_mode in (:sign, :unity, :dyson_sign, :dyson_unity) "distribution_mode must be :sign, :unity, :dyson_sign, or :dyson_unity"
    Γs = collect(Float64.(Γs))
    δ_values = collect(-δmax:dδ:δmax)
    p_values = collect(0.0:dp:pmax)
    resolved_output_dir = isnothing(output_dir) ?
        touching_wzw_output_dir(Γs, dδ, dp, δmax, pmax, distribution_mode) :
        String(output_dir)
    mkpath(resolved_output_dir)

    metadata = (
        Γs=Γs,
        dδ=dδ,
        δmax=δmax,
        dp=dp,
        pmax=pmax,
        η=η,
        distribution_mode=String(distribution_mode),
        pcut=pcut,
        ωcut=ωcut,
        cutoff_power=cutoff_power,
        δ_values=δ_values,
        p_values=p_values,
    )
    save_touching_wzw_metadata(resolved_output_dir, metadata)

    saved_files = String[]
    for Γ in Γs
        println("computing touching WZW stencil for Γ=", Γ)
        components = build_touching_patch_components(
            δ_values,
            p_values;
            Γ=Γ,
            η=η,
            distribution_mode=distribution_mode,
            pcut=pcut,
            ωcut=ωcut,
            cutoff_power=cutoff_power,
        )
        result = compute_wzw_stencil(components; dδ=dδ, dp=dp)
        filepath = save_touching_wzw_gamma(resolved_output_dir, Γ, result)
        println("saved ", filepath)
        push!(saved_files, filepath)
    end

    jldsave(joinpath(resolved_output_dir, "index.jld2"); output_dir=resolved_output_dir, Γs, saved_files)
    return (
        output_dir=resolved_output_dir,
        Γs=Γs,
        saved_files=saved_files,
    )
end

main(; kwargs...) = search_touching_wzw_cubic(; kwargs...)

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
