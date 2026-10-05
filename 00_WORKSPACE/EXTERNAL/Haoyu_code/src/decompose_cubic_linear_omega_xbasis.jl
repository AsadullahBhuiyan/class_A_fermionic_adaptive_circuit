using JLD2
using Printf
using Base.Threads: @threads, maxthreadid, threadid

include("collect_cubic_triangle_data.jl")

const XBASIS_CUBIC_DIR_PREFIX = "cubic_linear_omega_xbasis"

const CYCLE_LABELS = (
    "+++",
    "++-",
    "+-+",
    "+--",
    "-++",
    "-+-",
    "--+",
    "---",
)

const CYCLE_COMPONENTS = (
    (1, 1, 1),
    (1, 2, 3),
    (2, 3, 1),
    (2, 4, 3),
    (3, 1, 2),
    (3, 2, 4),
    (4, 3, 2),
    (4, 4, 4),
)

const SECTOR_LABELS = ("RAK", "KRA", "AKR", "KKK")

@inline function rotate_to_sigma_x_entries(
    g11::ComplexF64,
    g12::ComplexF64,
    g21::ComplexF64,
    g22::ComplexF64,
)
    gpp = 0.5 * (g11 + g12 + g21 + g22)
    gpm = 0.5 * (g11 - g12 + g21 - g22)
    gmp = 0.5 * (g11 + g12 - g21 - g22)
    gmm = 0.5 * (g11 - g12 - g21 + g22)
    return gpp, gpm, gmp, gmm
end

function build_green_components_xbasis(
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
            x11, x12, x21, x22 = rotate_to_sigma_x_entries(g11, g12, g21, g22)
            a11 = conj(x11)
            a12 = conj(x12)
            a21 = conj(x21)
            a22 = conj(x22)
            k11, k12, k21, k22 = keldysh_entries(
                mode,
                ωs[i],
                Γ,
                x11,
                x12,
                x21,
                x22,
                a11,
                a12,
                a21,
                a22,
            )

            GR[i, j, 1] = x11
            GR[i, j, 2] = x12
            GR[i, j, 3] = x21
            GR[i, j, 4] = x22

            GA[i, j, 1] = a11
            GA[i, j, 2] = a12
            GA[i, j, 3] = a21
            GA[i, j, 4] = a22

            GK[i, j, 1] = k11
            GK[i, j, 2] = k12
            GK[i, j, 3] = k21
            GK[i, j, 4] = k22
        end
    end

    return GR, GA, GK
end

@inline function cycle_term(
    A::Array{ComplexF64,3},
    iA::Int,
    jA::Int,
    B::Array{ComplexF64,3},
    iB::Int,
    jB::Int,
    C::Array{ComplexF64,3},
    iC::Int,
    jC::Int,
    cycle_idx::Int,
)
    ia, ib, ic = CYCLE_COMPONENTS[cycle_idx]
    return A[iA, jA, ia] * B[iB, jB, ib] * C[iC, jC, ic]
end

function decompose_cubic_linear_omega_for_gamma(
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
    Ω_shifts = centered_shifts(5)
    Ω_values = dω .* Ω_shifts
    zero_shift = findfirst(iszero, Ω_shifts)
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
    partial_Ω1 = zeros(ComplexF64, nt, length(Ω_shifts), length(SECTOR_LABELS), length(CYCLE_LABELS))
    partial_Ω2 = zeros(ComplexF64, nt, length(Ω_shifts), length(SECTOR_LABELS), length(CYCLE_LABELS))

    @threads for jq in 1:Nk
        tid = threadid()
        @inbounds for iω in 1:Nω
            for (sidx, n) in pairs(Ω_shifts)
                iωs = iω - n
                if 1 <= iωs <= Nω
                    for cycle_idx in eachindex(CYCLE_LABELS)
                        partial_Ω1[tid, sidx, 1, cycle_idx] += cycle_term(GR, iω, jq, GA, iωs, jq, GK, iωs, jq, cycle_idx)
                        partial_Ω1[tid, sidx, 2, cycle_idx] += cycle_term(GK, iω, jq, GR, iωs, jq, GA, iωs, jq, cycle_idx)
                        partial_Ω1[tid, sidx, 3, cycle_idx] += cycle_term(GA, iω, jq, GK, iωs, jq, GR, iωs, jq, cycle_idx)
                        partial_Ω1[tid, sidx, 4, cycle_idx] -= cycle_term(GK, iω, jq, GK, iωs, jq, GK, iωs, jq, cycle_idx)

                        partial_Ω2[tid, sidx, 1, cycle_idx] += cycle_term(GR, iω, jq, GA, iω, jq, GK, iωs, jq, cycle_idx)
                        partial_Ω2[tid, sidx, 2, cycle_idx] += cycle_term(GK, iω, jq, GR, iω, jq, GA, iωs, jq, cycle_idx)
                        partial_Ω2[tid, sidx, 3, cycle_idx] += cycle_term(GA, iω, jq, GK, iω, jq, GR, iωs, jq, cycle_idx)
                        partial_Ω2[tid, sidx, 4, cycle_idx] -= cycle_term(GK, iω, jq, GK, iω, jq, GK, iωs, jq, cycle_idx)
                    end
                end
            end
        end
    end

    Ω1_values = prefactor .* dropdims(sum(partial_Ω1; dims=1); dims=1)
    Ω2_values = prefactor .* dropdims(sum(partial_Ω2; dims=1); dims=1)

    ΔΩ = Ω_values[zero_shift + 1] - Ω_values[zero_shift]
    dΩ1 = (Ω1_values[zero_shift + 1, :, :] .- Ω1_values[zero_shift - 1, :, :]) ./ (2 * ΔΩ)
    dΩ2 = (Ω2_values[zero_shift + 1, :, :] .- Ω2_values[zero_shift - 1, :, :]) ./ (2 * ΔΩ)
    scaled_dΩ1 = Γ^3 .* dΩ1
    scaled_dΩ2 = Γ^3 .* dΩ2

    sector_sum_dΩ1 = dropdims(sum(scaled_dΩ1; dims=2); dims=2)
    sector_sum_dΩ2 = dropdims(sum(scaled_dΩ2; dims=2); dims=2)
    cycle_sum_dΩ1 = dropdims(sum(scaled_dΩ1; dims=1); dims=1)
    cycle_sum_dΩ2 = dropdims(sum(scaled_dΩ2; dims=1); dims=1)
    total_dΩ1 = sum(sector_sum_dΩ1)
    total_dΩ2 = sum(sector_sum_dΩ2)

    return (
        Γ=Γ,
        Ω_values=Ω_values,
        sector_labels=collect(SECTOR_LABELS),
        cycle_labels=collect(CYCLE_LABELS),
        Ω1_values=Ω1_values,
        Ω2_values=Ω2_values,
        scaled_dΩ1=scaled_dΩ1,
        scaled_dΩ2=scaled_dΩ2,
        sector_sum_dΩ1=sector_sum_dΩ1,
        sector_sum_dΩ2=sector_sum_dΩ2,
        cycle_sum_dΩ1=cycle_sum_dΩ1,
        cycle_sum_dΩ2=cycle_sum_dΩ2,
        total_dΩ1=total_dΩ1,
        total_dΩ2=total_dΩ2,
    )
end

function save_cubic_linear_omega_xbasis(output_dir::AbstractString, result::NamedTuple)
    filepath = joinpath(output_dir, "gamma_$(gamma_token(result.Γ)).jld2")
    jldsave(filepath; result...)
    return filepath
end

function print_cubic_linear_omega_summary(result::NamedTuple)
    println("Γ = ", result.Γ)
    println("Γ^3 dΩ1 total = ", result.total_dΩ1)
    println("Γ^3 dΩ2 total = ", result.total_dΩ2)
    println("sector sums for Ω1:")
    for i in eachindex(result.sector_labels)
        println("  ", result.sector_labels[i], ": ", result.sector_sum_dΩ1[i])
    end
    println("sector sums for Ω2:")
    for i in eachindex(result.sector_labels)
        println("  ", result.sector_labels[i], ": ", result.sector_sum_dΩ2[i])
    end
end

function collect_cubic_linear_omega_xbasis_breakdown(;
    Γs::AbstractVector=[20.0, 40.0, 80.0],
    Nω::Int=12800,
    Nk::Int=256,
    dω::Float64=0.05,
    r::Float64=1.0,
    η::Float64=1e-4,
    distribution_mode::Symbol=:dyson_sign,
    output_dir::Union{Nothing,AbstractString}=nothing,
)
    resolved_output_dir = isnothing(output_dir) ?
        "$(XBASIS_CUBIC_DIR_PREFIX)_$(String(distribution_mode))_Nw$(Nω)_Nk$(Nk)_dw$(replace(@sprintf("%.3f", dω), "." => "p"))" :
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
        println("computing sigma_x cubic linear-Ω breakdown for Γ=", Γ)
        result = decompose_cubic_linear_omega_for_gamma(
            Γ;
            Nω=Nω,
            Nk=Nk,
            dω=dω,
            r=r,
            η=η,
            distribution_mode=distribution_mode,
        )
        filepath = save_cubic_linear_omega_xbasis(resolved_output_dir, result)
        print_cubic_linear_omega_summary(result)
        push!(saved_files, filepath)
    end

    jldsave(joinpath(resolved_output_dir, "index.jld2"); output_dir=resolved_output_dir, saved_files)
    return (
        output_dir=resolved_output_dir,
        saved_files=saved_files,
    )
end

main(; kwargs...) = collect_cubic_linear_omega_xbasis_breakdown(; kwargs...)

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
