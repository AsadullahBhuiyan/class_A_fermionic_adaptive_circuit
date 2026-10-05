using JLD2
using Printf

include("estimate_cubic_mmm_dynamic_model.jl")

const MMM_WZW_DYNAMIC_DIR_PREFIX = "cubic_mmm_wzw_dynamic"

function compute_scalar_triangle_value(
    GR::Array{ComplexF64,2},
    GA::Array{ComplexF64,2},
    GK::Array{ComplexF64,2},
    n1::Int,
    m1::Int,
    n2::Int,
    m2::Int,
    dω::Float64,
)
    Nω, Nq = size(GR)
    prefactor = dω / (2π * Nq)
    acc = 0.0 + 0.0im

    @inbounds for jq in 1:Nq
        jq1 = mod1(jq - m1, Nq)
        jq2 = mod1(jq - m1 - m2, Nq)
        for iω in 1:Nω
            iω1 = iω - n1
            iω2 = iω - n1 - n2
            if !(1 <= iω1 <= Nω && 1 <= iω2 <= Nω)
                continue
            end
            gR0 = GR[iω, jq]
            gA0 = GA[iω, jq]
            gK0 = GK[iω, jq]
            gR1 = GR[iω1, jq1]
            gA1 = GA[iω1, jq1]
            gK1 = GK[iω1, jq1]
            gR2 = GR[iω2, jq2]
            gA2 = GA[iω2, jq2]
            gK2 = GK[iω2, jq2]
            acc += gR0 * gA1 * gK2 + gK0 * gR1 * gA2 + gA0 * gK1 * gR2 - gK0 * gK1 * gK2
        end
    end

    return prefactor * acc
end

function build_full_exact_mm_components(
    ωs::AbstractVector,
    qs::AbstractVector;
    r::Float64,
    Γ::Float64,
    η::Float64,
    distribution_mode::Symbol,
)
    full_GR, full_GA, full_GK = build_green_components_xbasis(
        ωs,
        qs;
        r=r,
        Γ=Γ,
        η=η,
        distribution_mode=distribution_mode,
    )
    return full_GR[:, :, 4], full_GA[:, :, 4], full_GK[:, :, 4]
end

function mixed_derivative(values::NTuple{4,ComplexF64}, hx::Float64, hy::Float64)
    fpp, fpm, fmp, fmm = values
    return (fpp - fpm - fmp + fmm) / (4 * hx * hy)
end

function compute_wzw_scalar_stencil(
    GR::Array{ComplexF64,2},
    GA::Array{ComplexF64,2},
    GK::Array{ComplexF64,2},
    dω::Float64,
    Δk::Float64,
)
    f_Ω1k2_pp = compute_scalar_triangle_value(GR, GA, GK, +1, 0, 0, +1, dω)
    f_Ω1k2_pm = compute_scalar_triangle_value(GR, GA, GK, +1, 0, 0, -1, dω)
    f_Ω1k2_mp = compute_scalar_triangle_value(GR, GA, GK, -1, 0, 0, +1, dω)
    f_Ω1k2_mm = compute_scalar_triangle_value(GR, GA, GK, -1, 0, 0, -1, dω)

    f_k1Ω2_pp = compute_scalar_triangle_value(GR, GA, GK, 0, +1, +1, 0, dω)
    f_k1Ω2_pm = compute_scalar_triangle_value(GR, GA, GK, 0, +1, -1, 0, dω)
    f_k1Ω2_mp = compute_scalar_triangle_value(GR, GA, GK, 0, -1, +1, 0, dω)
    f_k1Ω2_mm = compute_scalar_triangle_value(GR, GA, GK, 0, -1, -1, 0, dω)

    dΩ1dk2 = mixed_derivative((f_Ω1k2_pp, f_Ω1k2_pm, f_Ω1k2_mp, f_Ω1k2_mm), dω, Δk)
    dk1dΩ2 = mixed_derivative((f_k1Ω2_pp, f_k1Ω2_pm, f_k1Ω2_mp, f_k1Ω2_mm), Δk, dω)
    anti = 0.5 * (dΩ1dk2 - dk1dΩ2)
    sym = 0.5 * (dΩ1dk2 + dk1dΩ2)

    return (
        dΩ1dk2=dΩ1dk2,
        dk1dΩ2=dk1dΩ2,
        anti=anti,
        sym=sym,
    )
end

function collect_cubic_mmm_wzw_dynamic(;
    Γs::AbstractVector=[20.0, 40.0, 80.0, 160.0],
    Nω::Int=25_600,
    Nk::Int=256,
    dω::Float64=0.025,
    r::Float64=1.0,
    η::Float64=1e-4,
    output_dir::Union{Nothing,AbstractString}=nothing,
)
    dw_token = replace(@sprintf("%.4f", dω), "." => "p")
    resolved_output_dir = isnothing(output_dir) ?
        "data/$(MMM_WZW_DYNAMIC_DIR_PREFIX)_dyson_sign_Nw$(Nω)_Nk$(Nk)_dw$(dw_token)" :
        String(output_dir)
    mkpath(resolved_output_dir)

    ωs = physical_frequency_grid(Nω, dω)
    qs = collect(2π .* (0:(Nk - 1)) ./ Nk)
    Δk = 2π / Nk

    metadata = (
        Γs=collect(Float64.(Γs)),
        Nω=Nω,
        Nk=Nk,
        dω=dω,
        Δk=Δk,
        r=r,
        η=η,
    )
    jldsave(joinpath(resolved_output_dir, "metadata.jld2"); metadata...)

    report_path = joinpath(resolved_output_dir, "report.txt")
    open(report_path, "w") do io
        println(io, "Exact vs reduced dynamic --- WZW stencil")
        println(io)
        println(io, @sprintf("%6s  %22s  %22s  %22s  %22s", "Γ", "exact anti", "reduced anti", "Γ^3 exact anti", "Γ^3 reduced anti"))
        for Γraw in Γs
            Γ = Float64(Γraw)
            exact_GR, exact_GA, exact_GK = build_full_exact_mm_components(
                ωs,
                qs;
                r=r,
                Γ=Γ,
                η=η,
                distribution_mode=:dyson_sign,
            )
            reduced_GR, reduced_GA, reduced_GK = build_reduced_dynamic_mm_components(
                ωs,
                qs;
                r=r,
                Γ=Γ,
                η=η,
            )

            exact = compute_wzw_scalar_stencil(exact_GR, exact_GA, exact_GK, dω, Δk)
            reduced = compute_wzw_scalar_stencil(reduced_GR, reduced_GA, reduced_GK, dω, Δk)

            jldsave(
                joinpath(resolved_output_dir, "gamma_$(gamma_token(Γ)).jld2");
                Γ,
                exact_dΩ1dk2=exact.dΩ1dk2,
                exact_dk1dΩ2=exact.dk1dΩ2,
                exact_anti=exact.anti,
                exact_sym=exact.sym,
                reduced_dΩ1dk2=reduced.dΩ1dk2,
                reduced_dk1dΩ2=reduced.dk1dΩ2,
                reduced_anti=reduced.anti,
                reduced_sym=reduced.sym,
            )

            println(
                io,
                @sprintf(
                    "%6.1f  %22.15e  %22.15e  %22.15e  %22.15e",
                    Γ,
                    real(exact.anti),
                    real(reduced.anti),
                    real(Γ^3 * exact.anti),
                    real(Γ^3 * reduced.anti),
                ),
            )
        end
    end

    return resolved_output_dir
end

function main()
    out = collect_cubic_mmm_wzw_dynamic()
    println("saved --- WZW dynamic comparison to ", out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
