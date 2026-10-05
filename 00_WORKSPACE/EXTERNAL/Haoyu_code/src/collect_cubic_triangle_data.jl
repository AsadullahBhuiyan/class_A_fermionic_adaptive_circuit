using JLD2
using LinearAlgebra
using Printf
using Base.Threads: @threads, nthreads

include("quadratic_kernel.jl")

const CUBIC_TRIANGLE_SCAN_DIR_PREFIX = "cubic_triangle_scan"

function gamma_token(Γ::Real)
    return replace(@sprintf("%.3f", float(Γ)), "." => "p")
end

function omega_token(ωmax::Real)
    return replace(@sprintf("%.3f", float(ωmax)), "." => "p")
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

@inline function keldysh_entries(
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

@inline function keldysh_entries(
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

@inline function keldysh_entries(
    mode::Union{Val{:dyson_sign}, Val{:dyson_unity}},
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

function cubic_scan_output_dir(Nω::Int, Nk::Int, distribution_mode::Symbol, ωband::Float64)
    return "$(CUBIC_TRIANGLE_SCAN_DIR_PREFIX)_$(String(distribution_mode))_Nw$(Nω)_Nk$(Nk)_wmax$(omega_token(ωband))"
end

function centered_shifts(n::Int)
    @assert isodd(n) "external window size must be odd so that zero is included"
    radius = fld(n, 2)
    return collect(-radius:radius)
end

function physical_frequency_grid(Nω::Int, dω::Float64)
    νs = sort(vcat(collect(0:fld(Nω - 1, 2)), collect(-fld(Nω, 2):-1)))
    return dω .* νs
end

@inline function monitored_retarded_green_entries(ω::Float64, k::Float64, r::Float64, Γ::Float64, η::Float64)
    z = ComplexF64(ω, η)
    sk = sin(k)
    ck = cos(k)
    mass = r - ck
    Σ = Σbulk(z, k, r)
    shift = im * Γ / 2

    a = z - mass - Σ / 2 + shift
    b = -sk + Σ / 2
    d = z + mass - Σ / 2 + shift
    detinv = inv(a * d - b * b)

    return d * detinv, -b * detinv, -b * detinv, a * detinv
end

function build_green_components(
    ωs::AbstractVector,
    ks::AbstractVector;
    r::Float64,
    Γ::Float64,
    η::Float64,
    distribution_mode::Symbol,
)
    mode = Val(distribution_mode)
    Nω = length(ωs)
    Nk = length(ks)
    GR = Array{ComplexF64}(undef, Nω, Nk, 4)
    GA = Array{ComplexF64}(undef, Nω, Nk, 4)
    GK = Array{ComplexF64}(undef, Nω, Nk, 4)

    @inbounds for j in eachindex(ks)
        k = ks[j]
        for i in eachindex(ωs)
            g11, g12, g21, g22 = monitored_retarded_green_entries(ωs[i], k, r, Γ, η)
            ga11 = conj(g11)
            ga12 = conj(g12)
            ga21 = conj(g21)
            ga22 = conj(g22)
            k11, k12, k21, k22 = keldysh_entries(
                mode,
                ωs[i],
                Γ,
                g11,
                g12,
                g21,
                g22,
                ga11,
                ga12,
                ga21,
                ga22,
            )

            GR[i, j, 1] = g11
            GR[i, j, 2] = g12
            GR[i, j, 3] = g21
            GR[i, j, 4] = g22

            GA[i, j, 1] = ga11
            GA[i, j, 2] = ga12
            GA[i, j, 3] = ga21
            GA[i, j, 4] = ga22

            GK[i, j, 1] = k11
            GK[i, j, 2] = k12
            GK[i, j, 3] = k21
            GK[i, j, 4] = k22
        end
    end

    return GR, GA, GK
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

@inline function decode_external_index(idx::Int, NΩ::Int, Nkext::Int)
    t = idx - 1
    iΩ1 = mod(t, NΩ) + 1
    t ÷= NΩ
    ik1 = mod(t, Nkext) + 1
    t ÷= Nkext
    iΩ2 = mod(t, NΩ) + 1
    t ÷= NΩ
    ik2 = t + 1
    return iΩ1, ik1, iΩ2, ik2
end

function compute_cubic_triangle_grid(
    GR::Array{ComplexF64,3},
    GA::Array{ComplexF64,3},
    GK::Array{ComplexF64,3},
    Ω_shifts::AbstractVector{Int},
    k_shifts::AbstractVector{Int},
    dω::Float64,
)
    Nω = size(GR, 1)
    Nk = size(GR, 2)
    NΩ = length(Ω_shifts)
    Nkext = length(k_shifts)
    prefactor = dω / (2π * Nk)
    out = Array{ComplexF64}(undef, NΩ, Nkext, NΩ, Nkext)
    ncombo = length(out)

    @threads for linear_idx in 1:ncombo
        iΩ1, ik1, iΩ2, ik2 = decode_external_index(linear_idx, NΩ, Nkext)
        n1 = Ω_shifts[iΩ1]
        m1 = k_shifts[ik1]
        n2 = Ω_shifts[iΩ2]
        m2 = k_shifts[ik2]
        n12 = n1 + n2
        m12 = m1 + m2

        acc = 0.0 + 0.0im
        @inbounds for jq in 1:Nk
            jq1 = mod1(jq - m1, Nk)
            jq2 = mod1(jq - m12, Nk)
            for iω in 1:Nω
                iω1 = iω - n1
                iω2 = iω - n12
                if !(1 <= iω1 <= Nω && 1 <= iω2 <= Nω)
                    continue
                end

                acc += trace_product(GR, iω, jq, GA, iω1, jq1, GK, iω2, jq2)
                acc += trace_product(GK, iω, jq, GR, iω1, jq1, GA, iω2, jq2)
                acc += trace_product(GA, iω, jq, GK, iω1, jq1, GR, iω2, jq2)
                acc -= trace_product(GK, iω, jq, GK, iω1, jq1, GK, iω2, jq2)
            end
        end

        out[iΩ1, ik1, iΩ2, ik2] = prefactor * acc
    end

    return out
end

function save_cubic_scan_metadata(output_dir::AbstractString, metadata::NamedTuple)
    jldsave(joinpath(output_dir, "metadata.jld2"); metadata...)
    return nothing
end

function save_cubic_scan_gamma(
    output_dir::AbstractString,
    Γ::Float64,
    Ω_values::AbstractVector,
    k_values::AbstractVector,
    Ω_shifts::AbstractVector,
    k_shifts::AbstractVector,
    triangle::Array{ComplexF64,4},
)
    filepath = joinpath(output_dir, "gamma_$(gamma_token(Γ)).jld2")
    zero_Ω = cld(length(Ω_values), 2)
    zero_k = cld(length(k_values), 2)
    jldsave(
        filepath;
        Γ,
        Ω_values,
        k_values,
        Ω_shifts=collect(Ω_shifts),
        k_shifts=collect(k_shifts),
        triangle,
        center_value=triangle[zero_Ω, zero_k, zero_Ω, zero_k],
        Ω1_zero_slice=triangle[zero_Ω, zero_k, :, :],
        Ω2_zero_slice=triangle[:, :, zero_Ω, zero_k],
    )
    return filepath
end

function collect_cubic_triangle_data(;
    Nω::Int=3072,
    Nk::Int=256,
    dω::Float64=0.1,
    r::Float64=1.0,
    η::Float64=1e-4,
    Γs::AbstractVector=[20.0, 40.0, 80.0],
    NΩ_save::Int=5,
    Nk_save::Int=3,
    distribution_mode::Symbol=:dyson_sign,
    output_dir::Union{Nothing, AbstractString}=nothing,
)
    @assert distribution_mode in (:sign, :unity, :dyson_sign, :dyson_unity) "distribution_mode must be :sign, :unity, :dyson_sign, or :dyson_unity"
    @assert NΩ_save <= Nω "frequency window cannot exceed the physical frequency grid"
    @assert Nk_save <= Nk "momentum window cannot exceed the lattice grid"

    ω_phys = physical_frequency_grid(Nω, dω)
    ks = collect(2π .* (0:(Nk - 1)) ./ Nk)
    ωband = maximum(abs.(ω_phys))
    resolved_output_dir = isnothing(output_dir) ? cubic_scan_output_dir(Nω, Nk, distribution_mode, ωband) : String(output_dir)
    mkpath(resolved_output_dir)
    Ω_shifts = centered_shifts(NΩ_save)
    k_shifts = centered_shifts(Nk_save)
    Ω_values = dω .* Ω_shifts
    k_values = (2π / Nk) .* k_shifts

    metadata = (
        Nω=Nω,
        Nk=Nk,
        dω=dω,
        r=r,
        η=η,
        Γs=collect(Float64.(Γs)),
        ω_phys=ω_phys,
        ωband=ωband,
        ks=ks,
        Ω_shifts=collect(Ω_shifts),
        k_shifts=collect(k_shifts),
        Ω_values=Ω_values,
        k_values=k_values,
        distribution_mode=String(distribution_mode),
        nthreads=nthreads(),
    )
    save_cubic_scan_metadata(resolved_output_dir, metadata)

    saved_files = String[]
    for Γ in Float64.(Γs)
        println("computing cubic triangle scan for Γ=", Γ, " with distribution=", distribution_mode)
        GR, GA, GK = build_green_components(
            ω_phys,
            ks;
            r=r,
            Γ=Γ,
            η=η,
            distribution_mode=distribution_mode,
        )
        triangle = compute_cubic_triangle_grid(GR, GA, GK, Ω_shifts, k_shifts, dω)
        filepath = save_cubic_scan_gamma(
            resolved_output_dir,
            Γ,
            Ω_values,
            k_values,
            Ω_shifts,
            k_shifts,
            triangle,
        )
        println("saved ", filepath)
        push!(saved_files, filepath)
    end

    jldsave(
        joinpath(resolved_output_dir, "index.jld2");
        output_dir=resolved_output_dir,
        Γs=collect(Float64.(Γs)),
        saved_files,
        ωband,
        dω,
    )
    return (
        output_dir=resolved_output_dir,
        Γs=collect(Float64.(Γs)),
        Ω_values=Ω_values,
        k_values=k_values,
        ωband=ωband,
        saved_files=saved_files,
    )
end

main(; kwargs...) = collect_cubic_triangle_data(; kwargs...)

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
