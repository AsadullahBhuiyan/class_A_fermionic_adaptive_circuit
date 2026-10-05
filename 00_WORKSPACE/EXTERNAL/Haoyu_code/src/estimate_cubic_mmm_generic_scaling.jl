using JLD2
using Printf
using Statistics
using Base.Threads: @threads, maxthreadid, threadid

include("estimate_cubic_mmm_dynamic_model.jl")

const MMM_GENERIC_DIR_PREFIX = "cubic_mmm_generic_scaling"

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
    n12 = n1 + n2
    m12 = m1 + m2
    prefactor = dω / (2π * Nq)
    nt = maxthreadid()
    partial = zeros(ComplexF64, nt)

    @threads for jq in 1:Nq
        tid = threadid()
        jq1 = mod1(jq - m1, Nq)
        jq2 = mod1(jq - m12, Nq)
        acc = 0.0 + 0.0im
        @inbounds for iω in 1:Nω
            iω1 = iω - n1
            iω2 = iω - n12
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
        partial[tid] += acc
    end

    return prefactor * sum(partial)
end

function fit_log_cutoff_line(
    Ω_values::AbstractVector{Float64},
    values::AbstractVector{ComplexF64},
    Γ::Float64,
    k::Float64,
)
    y = imag.(values)
    x1 = Ω_values .* log.(1.0 ./ (abs.(Ω_values) .+ k^2))
    x2 = Ω_values
    X = hcat(x1, x2)
    coeff = X \ y
    fitted = X * coeff
    rel_rms = sqrt(sum(abs2, y .- fitted) / length(y)) / max(maximum(abs.(y)), eps())
    return (
        coeff_log=coeff[1],
        coeff_linear=coeff[2],
        rel_rms=rel_rms,
        scaled_log=-(Γ^4) * coeff[1],
        scaled_linear=-(Γ^4) * coeff[2],
    )
end

function collect_family_lines(
    GR::Array{ComplexF64,2},
    GA::Array{ComplexF64,2},
    GK::Array{ComplexF64,2},
    Ω_shifts::AbstractVector{Int},
    k_shifts::AbstractVector{Int},
    dω::Float64,
)
    NΩ = length(Ω_shifts)
    Nkext = length(k_shifts)
    family_A = Array{ComplexF64}(undef, NΩ, Nkext)
    family_B = Array{ComplexF64}(undef, NΩ, Nkext)

    @threads for idx in 1:(NΩ * Nkext)
        iΩ = mod(idx - 1, NΩ) + 1
        ik = (idx - 1) ÷ NΩ + 1
        n = Ω_shifts[iΩ]
        m = k_shifts[ik]
        family_A[iΩ, ik] = compute_scalar_triangle_value(GR, GA, GK, n, m, 0, 0, dω)
        family_B[iΩ, ik] = compute_scalar_triangle_value(GR, GA, GK, n, 0, 0, m, dω)
    end

    return family_A, family_B
end

function collect_cubic_mmm_generic_scaling(;
    Γ::Float64=80.0,
    Nω::Int=25_600,
    Nk::Int=256,
    dω::Float64=0.0125,
    Ω_shifts::AbstractVector{Int}=[-8, -4, -2, -1, 1, 2, 4, 8],
    k_shifts::AbstractVector{Int}=[1, 2, 4],
    r::Float64=1.0,
    η::Float64=1e-4,
    output_dir::Union{Nothing,AbstractString}=nothing,
)
    dw_token = replace(@sprintf("%.4f", dω), "." => "p")
    resolved_output_dir = isnothing(output_dir) ?
        "$(MMM_GENERIC_DIR_PREFIX)_dyson_sign_gamma$(gamma_token(Γ))_Nw$(Nω)_Nk$(Nk)_dw$(dw_token)" :
        String(output_dir)
    mkpath(resolved_output_dir)

    ωs = physical_frequency_grid(Nω, dω)
    qs = collect(2π .* (0:(Nk - 1)) ./ Nk)
    Ω_values = dω .* Float64.(Ω_shifts)
    k_values = (2π / Nk) .* Float64.(k_shifts)

    exact_GR, exact_GA, exact_GK = build_exact_mm_components(
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

    exact_A, exact_B = collect_family_lines(exact_GR, exact_GA, exact_GK, Ω_shifts, k_shifts, dω)
    reduced_A, reduced_B = collect_family_lines(reduced_GR, reduced_GA, reduced_GK, Ω_shifts, k_shifts, dω)

    metadata = (
        Γ=Γ,
        Nω=Nω,
        Nk=Nk,
        dω=dω,
        Ω_shifts=collect(Int.(Ω_shifts)),
        Ω_values=Ω_values,
        k_shifts=collect(Int.(k_shifts)),
        k_values=k_values,
        r=r,
        η=η,
    )
    jldsave(joinpath(resolved_output_dir, "metadata.jld2"); metadata...)
    jldsave(
        joinpath(resolved_output_dir, "lines.jld2");
        exact_A,
        exact_B,
        reduced_A,
        reduced_B,
        Ω_values,
        k_values,
    )

    report_path = joinpath(resolved_output_dir, "report.txt")
    open(report_path, "w") do io
        println(io, "Γ = ", Γ)
        println(io, "Generic --- scaling scan")
        println(io, "family A: Q1 = (Ω, k), Q2 = (0, 0)")
        println(io, "family B: Q1 = (Ω, 0), Q2 = (0, k)")
        println(io)

        pos_sel = findall(>(0.0), Ω_values)
        for (ik, k) in pairs(k_values)
            exact_fit_A = fit_log_cutoff_line(Ω_values[pos_sel], exact_A[pos_sel, ik], Γ, abs(k))
            reduced_fit_A = fit_log_cutoff_line(Ω_values[pos_sel], reduced_A[pos_sel, ik], Γ, abs(k))
            exact_fit_B = fit_log_cutoff_line(Ω_values[pos_sel], exact_B[pos_sel, ik], Γ, abs(k))
            reduced_fit_B = fit_log_cutoff_line(Ω_values[pos_sel], reduced_B[pos_sel, ik], Γ, abs(k))

            println(io, "k = ", k)
            println(io, "  family A exact  : scaled log = ", exact_fit_A.scaled_log, ", scaled linear = ", exact_fit_A.scaled_linear, ", rel_rms = ", exact_fit_A.rel_rms)
            println(io, "  family A reduced: scaled log = ", reduced_fit_A.scaled_log, ", scaled linear = ", reduced_fit_A.scaled_linear, ", rel_rms = ", reduced_fit_A.rel_rms)
            println(io, "  family B exact  : scaled log = ", exact_fit_B.scaled_log, ", scaled linear = ", exact_fit_B.scaled_linear, ", rel_rms = ", exact_fit_B.rel_rms)
            println(io, "  family B reduced: scaled log = ", reduced_fit_B.scaled_log, ", scaled linear = ", reduced_fit_B.scaled_linear, ", rel_rms = ", reduced_fit_B.rel_rms)
            println(io)
        end
    end

    return resolved_output_dir
end

function main()
    collect_cubic_mmm_generic_scaling()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
