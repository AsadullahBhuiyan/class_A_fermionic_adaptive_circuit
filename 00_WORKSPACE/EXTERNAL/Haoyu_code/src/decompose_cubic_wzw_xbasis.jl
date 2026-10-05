using JLD2
using Printf
using Base.Threads: @threads, maxthreadid, threadid

include("decompose_cubic_linear_omega_xbasis.jl")

const XBASIS_CUBIC_WZW_DIR_PREFIX = "cubic_wzw_xbasis"

function accumulate_stencil!(
    partial::Array{ComplexF64,4},
    GR::Array{ComplexF64,3},
    GA::Array{ComplexF64,3},
    GK::Array{ComplexF64,3},
    n1::Int,
    m1::Int,
    n2::Int,
    m2::Int,
    slot::Int,
)
    Nω = size(GR, 1)
    Nk = size(GR, 2)

    @threads for jq in 1:Nk
        tid = threadid()
        jq1 = mod1(jq - m1, Nk)
        jq2 = mod1(jq - m1 - m2, Nk)
        @inbounds for iω in 1:Nω
            iω1 = iω - n1
            iω2 = iω - n1 - n2
            if !(1 <= iω1 <= Nω && 1 <= iω2 <= Nω)
                continue
            end
            for cycle_idx in eachindex(CYCLE_LABELS)
                partial[tid, slot, 1, cycle_idx] += cycle_term(GR, iω, jq, GA, iω1, jq1, GK, iω2, jq2, cycle_idx)
                partial[tid, slot, 2, cycle_idx] += cycle_term(GK, iω, jq, GR, iω1, jq1, GA, iω2, jq2, cycle_idx)
                partial[tid, slot, 3, cycle_idx] += cycle_term(GA, iω, jq, GK, iω1, jq1, GR, iω2, jq2, cycle_idx)
                partial[tid, slot, 4, cycle_idx] -= cycle_term(GK, iω, jq, GK, iω1, jq1, GK, iω2, jq2, cycle_idx)
            end
        end
    end
    return nothing
end

function mixed_derivative(values::NTuple{4,ComplexF64}, hx::Float64, hy::Float64)
    fpp, fpm, fmp, fmm = values
    return (fpp - fpm - fmp + fmm) / (4 * hx * hy)
end

function decompose_cubic_wzw_for_gamma(
    Γ::Float64;
    Nω::Int,
    Nk::Int,
    dω::Float64,
    r::Float64,
    η::Float64,
    distribution_mode::Symbol,
)
    ωs = physical_frequency_grid(Nω, dω)
    ks = collect(2π .* (0:(Nk - 1)) ./ Nk)
    Δk = 2π / Nk
    prefactor = dω / (2π * Nk)

    GR, GA, GK = build_green_components_xbasis(
        ωs,
        ks;
        r=r,
        Γ=Γ,
        η=η,
        distribution_mode=distribution_mode,
    )

    nt = maxthreadid()
    partial = zeros(ComplexF64, nt, 8, length(SECTOR_LABELS), length(CYCLE_LABELS))

    # Ω1-k2 stencil
    accumulate_stencil!(partial, GR, GA, GK, +1, 0, 0, +1, 1)
    accumulate_stencil!(partial, GR, GA, GK, +1, 0, 0, -1, 2)
    accumulate_stencil!(partial, GR, GA, GK, -1, 0, 0, +1, 3)
    accumulate_stencil!(partial, GR, GA, GK, -1, 0, 0, -1, 4)

    # k1-Ω2 stencil
    accumulate_stencil!(partial, GR, GA, GK, 0, +1, +1, 0, 5)
    accumulate_stencil!(partial, GR, GA, GK, 0, +1, -1, 0, 6)
    accumulate_stencil!(partial, GR, GA, GK, 0, -1, +1, 0, 7)
    accumulate_stencil!(partial, GR, GA, GK, 0, -1, -1, 0, 8)

    stencil_values = prefactor .* dropdims(sum(partial; dims=1); dims=1)

    dΩ1dk2 = similar(stencil_values, ComplexF64, length(SECTOR_LABELS), length(CYCLE_LABELS))
    dk1dΩ2 = similar(stencil_values, ComplexF64, length(SECTOR_LABELS), length(CYCLE_LABELS))
    anti = similar(stencil_values, ComplexF64, length(SECTOR_LABELS), length(CYCLE_LABELS))
    sym = similar(stencil_values, ComplexF64, length(SECTOR_LABELS), length(CYCLE_LABELS))

    @inbounds for isector in eachindex(SECTOR_LABELS), icycle in eachindex(CYCLE_LABELS)
        dΩ1dk2[isector, icycle] = mixed_derivative((
            stencil_values[1, isector, icycle],
            stencil_values[2, isector, icycle],
            stencil_values[3, isector, icycle],
            stencil_values[4, isector, icycle],
        ), dω, Δk)
        dk1dΩ2[isector, icycle] = mixed_derivative((
            stencil_values[5, isector, icycle],
            stencil_values[6, isector, icycle],
            stencil_values[7, isector, icycle],
            stencil_values[8, isector, icycle],
        ), Δk, dω)
        anti[isector, icycle] = 0.5 * (dΩ1dk2[isector, icycle] - dk1dΩ2[isector, icycle])
        sym[isector, icycle] = 0.5 * (dΩ1dk2[isector, icycle] + dk1dΩ2[isector, icycle])
    end

    cycle_sum_anti = dropdims(sum(anti; dims=1); dims=1)
    sector_sum_anti = dropdims(sum(anti; dims=2); dims=2)
    cycle_sum_sym = dropdims(sum(sym; dims=1); dims=1)
    sector_sum_sym = dropdims(sum(sym; dims=2); dims=2)

    return (
        Γ=Γ,
        dω=dω,
        Δk=Δk,
        sector_labels=collect(SECTOR_LABELS),
        cycle_labels=collect(CYCLE_LABELS),
        stencil_labels=collect((
            "Ω1k2_pp", "Ω1k2_pm", "Ω1k2_mp", "Ω1k2_mm",
            "k1Ω2_pp", "k1Ω2_pm", "k1Ω2_mp", "k1Ω2_mm",
        )),
        stencil_values=stencil_values,
        dΩ1dk2=dΩ1dk2,
        dk1dΩ2=dk1dΩ2,
        anti=anti,
        sym=sym,
        cycle_sum_anti=cycle_sum_anti,
        sector_sum_anti=sector_sum_anti,
        cycle_sum_sym=cycle_sum_sym,
        sector_sum_sym=sector_sum_sym,
        total_anti=sum(cycle_sum_anti),
        total_sym=sum(cycle_sum_sym),
    )
end

function save_cubic_wzw_xbasis(output_dir::AbstractString, result::NamedTuple)
    filepath = joinpath(output_dir, "gamma_$(gamma_token(result.Γ)).jld2")
    jldsave(filepath; result...)
    return filepath
end

function print_cubic_wzw_summary(result::NamedTuple)
    println("Γ = ", result.Γ)
    println("anti total = ", result.total_anti)
    println("sym total  = ", result.total_sym)
    println("cycle sums for anti:")
    for i in eachindex(result.cycle_labels)
        println("  ", result.cycle_labels[i], ": ", result.cycle_sum_anti[i])
    end
end

function collect_cubic_wzw_xbasis_breakdown(;
    Γs::AbstractVector=[20.0, 40.0, 80.0],
    Nω::Int=12_800,
    Nk::Int=256,
    dω::Float64=0.05,
    r::Float64=1.0,
    η::Float64=1e-4,
    distribution_mode::Symbol=:dyson_sign,
    output_dir::Union{Nothing,AbstractString}=nothing,
)
    resolved_output_dir = isnothing(output_dir) ?
        "data/$(XBASIS_CUBIC_WZW_DIR_PREFIX)_$(String(distribution_mode))_Nw$(Nω)_Nk$(Nk)_dw$(replace(@sprintf("%.3f", dω), "." => "p"))" :
        String(output_dir)
    mkpath(resolved_output_dir)

    metadata = (
        Γs=collect(Float64.(Γs)),
        Nω=Nω,
        Nk=Nk,
        dω=dω,
        r=r,
        η=η,
        distribution_mode=String(distribution_mode),
        sector_labels=collect(SECTOR_LABELS),
        cycle_labels=collect(CYCLE_LABELS),
    )
    jldsave(joinpath(resolved_output_dir, "metadata.jld2"); metadata...)

    saved_files = String[]
    for Γ in Float64.(Γs)
        println("computing sigma_x cubic WZW breakdown for Γ=", Γ)
        result = decompose_cubic_wzw_for_gamma(
            Γ;
            Nω=Nω,
            Nk=Nk,
            dω=dω,
            r=r,
            η=η,
            distribution_mode=distribution_mode,
        )
        filepath = save_cubic_wzw_xbasis(resolved_output_dir, result)
        print_cubic_wzw_summary(result)
        push!(saved_files, filepath)
    end

    jldsave(joinpath(resolved_output_dir, "index.jld2"); output_dir=resolved_output_dir, saved_files)
    return (
        output_dir=resolved_output_dir,
        saved_files=saved_files,
    )
end

main(; kwargs...) = collect_cubic_wzw_xbasis_breakdown(; kwargs...)

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
