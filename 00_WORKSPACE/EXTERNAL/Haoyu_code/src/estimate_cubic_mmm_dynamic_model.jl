using JLD2
using LinearAlgebra
using Printf
using Base.Threads: @threads, maxthreadid, threadid

include("decompose_cubic_linear_omega_xbasis.jl")

const MMM_DYNAMIC_DIR_PREFIX = "cubic_mmm_dynamic_line"

@inline function q_to_signed(q::Float64)
    return q <= π ? q : q - 2π
end

function build_exact_mm_components(
    ωs::AbstractVector,
    qs::AbstractVector;
    r::Float64,
    Γ::Float64,
    η::Float64,
    distribution_mode::Symbol,
)
    mode = Val(distribution_mode)
    Nω = length(ωs)
    Nq = length(qs)
    GR = Array{ComplexF64}(undef, Nω, Nq)
    GA = Array{ComplexF64}(undef, Nω, Nq)
    GK = Array{ComplexF64}(undef, Nω, Nq)

    @inbounds for jq in eachindex(qs)
        q = qs[jq]
        for iω in eachindex(ωs)
            g11, g12, g21, g22 = monitored_retarded_green_entries(ωs[iω], q, r, Γ, η)
            _, _, _, x22 = rotate_to_sigma_x_entries(g11, g12, g21, g22)
            ga22 = conj(x22)
            _, _, _, k22 = keldysh_entries(
                mode,
                ωs[iω],
                Γ,
                0.0 + 0.0im,
                0.0 + 0.0im,
                0.0 + 0.0im,
                x22,
                0.0 + 0.0im,
                0.0 + 0.0im,
                0.0 + 0.0im,
                ga22,
            )
            GR[iω, jq] = x22
            GA[iω, jq] = ga22
            GK[iω, jq] = k22
        end
    end

    return GR, GA, GK
end

function build_reduced_dynamic_mm_components(
    ωs::AbstractVector,
    qs::AbstractVector;
    r::Float64,
    Γ::Float64,
    η::Float64,
)
    Nω = length(ωs)
    Nq = length(qs)
    GR = Array{ComplexF64}(undef, Nω, Nq)
    GA = Array{ComplexF64}(undef, Nω, Nq)
    GK = Array{ComplexF64}(undef, Nω, Nq)
    @inbounds for jq in eachindex(qs)
        q = qs[jq]
        sq = sin(q)
        for iω in eachindex(ωs)
            ω = ωs[iω]
            z = ComplexF64(ω, η)
            Σ = Σbulk(z, q, r)
            d = ComplexF64(ω + sq, Γ / 2) - Σ
            gR = inv(d)
            gA = conj(gR)
            β = imag(Σ)
            gK = 2im * distribution_value(Val(:sign), ω) * β * gR * gA

            GR[iω, jq] = gR
            GA[iω, jq] = gA
            GK[iω, jq] = gK
        end
    end

    return GR, GA, GK
end

function compute_scalar_cycle_line(
    GR::Array{ComplexF64,2},
    GA::Array{ComplexF64,2},
    GK::Array{ComplexF64,2},
    shifts::AbstractVector{Int},
    dω::Float64,
)
    Nω, Nq = size(GR)
    prefactor = dω / (2π * Nq)
    nt = maxthreadid()
    partial = zeros(ComplexF64, nt, length(shifts))

    @threads for jq in 1:Nq
        tid = threadid()
        @inbounds for iω in 1:Nω
            gR = GR[iω, jq]
            gA = GA[iω, jq]
            gK = GK[iω, jq]
            for (sidx, n) in pairs(shifts)
                iωs = iω - n
                if 1 <= iωs <= Nω
                    hR = GR[iωs, jq]
                    hA = GA[iωs, jq]
                    hK = GK[iωs, jq]
                    partial[tid, sidx] += gR * hA * hK + gK * hR * hA + gA * hK * hR - gK * hK * hK
                end
            end
        end
    end

    return prefactor .* vec(sum(partial; dims=1))
end

function fit_odd_omega_log_line(Ω_positive::AbstractVector{Float64}, values::AbstractVector{ComplexF64})
    y = imag.(values)
    x1 = Ω_positive .* log.(1.0 ./ Ω_positive)
    x2 = Ω_positive
    X = hcat(x1, x2)
    coeff = X \ y
    fitted = X * coeff
    rel_rms = sqrt(sum(abs2, y .- fitted) / length(y)) / max(maximum(abs.(y)), eps())
    return (
        coeff_log=coeff[1],
        coeff_linear=coeff[2],
        fitted=fitted,
        rel_rms=rel_rms,
    )
end

function qresolved_cycle_difference(
    GR::Array{ComplexF64,2},
    GA::Array{ComplexF64,2},
    GK::Array{ComplexF64,2},
    shift::Int,
    dω::Float64,
)
    Nω, Nq = size(GR)
    prefactor = dω / (2π * Nq)
    out = zeros(ComplexF64, Nq)

    @inbounds for jq in 1:Nq
        for iω in 1:Nω
            iωp = iω - shift
            iωm = iω + shift
            if 1 <= iωp <= Nω
                gR = GR[iω, jq]
                gA = GA[iω, jq]
                gK = GK[iω, jq]
                hR = GR[iωp, jq]
                hA = GA[iωp, jq]
                hK = GK[iωp, jq]
                out[jq] += gR * hA * hK + gK * hR * hA + gA * hK * hR - gK * hK * hK
            end
            if 1 <= iωm <= Nω
                gR = GR[iω, jq]
                gA = GA[iω, jq]
                gK = GK[iω, jq]
                hR = GR[iωm, jq]
                hA = GA[iωm, jq]
                hK = GK[iωm, jq]
                out[jq] -= gR * hA * hK + gK * hR * hA + gA * hK * hR - gK * hK * hK
            end
        end
    end

    return prefactor .* out
end

function write_dynamic_report(
    output_dir::AbstractString,
    Γ::Float64,
    Ω_values::AbstractVector{Float64},
    exact_values::AbstractVector{ComplexF64},
    reduced_values::AbstractVector{ComplexF64},
    exact_fit,
    reduced_fit,
    q_peaks::AbstractVector,
)
    path = joinpath(output_dir, "report_gamma_$(gamma_token(Γ)).txt")
    open(path, "w") do io
        println(io, "Γ = ", Γ)
        println(io, "Ω-line for the --- cycle")
        println(io)
        println(io, @sprintf("%8s  %22s  %22s", "Ω", "exact Im", "reduced Im"))
        for i in eachindex(Ω_values)
            println(
                io,
                @sprintf(
                    "%8.4f  %22.15e  %22.15e",
                    Ω_values[i],
                    imag(exact_values[i]),
                    imag(reduced_values[i]),
                ),
            )
        end
        println(io)
        println(io, "odd Ω log(1/Ω) + Ω fit, exact:")
        println(io, "  coeff_log    = ", exact_fit.coeff_log)
        println(io, "  coeff_linear = ", exact_fit.coeff_linear)
        println(io, "  rel_rms      = ", exact_fit.rel_rms)
        println(io)
        println(io, "odd Ω log(1/Ω) + Ω fit, reduced dynamic:")
        println(io, "  coeff_log    = ", reduced_fit.coeff_log)
        println(io, "  coeff_linear = ", reduced_fit.coeff_linear)
        println(io, "  rel_rms      = ", reduced_fit.rel_rms)
        println(io)
        println(io, "largest exact |Im| q-peaks for Ω = ±", Ω_values[findfirst(>(0.0), Ω_values)])
        for (q, value) in q_peaks
            println(io, "  q = ", q, "  value = ", value)
        end
    end
    return path
end

function collect_cubic_mmm_dynamic_line(;
    Γs::AbstractVector=[40.0, 80.0],
    Nω::Int=25_600,
    Nk::Int=256,
    dω::Float64=0.025,
    Ω_shifts::AbstractVector{Int}=[-8, -4, -2, -1, 0, 1, 2, 4, 8],
    r::Float64=1.0,
    η::Float64=1e-4,
    output_dir::Union{Nothing,AbstractString}=nothing,
)
    dw_token = replace(@sprintf("%.3f", dω), "." => "p")
    resolved_output_dir = isnothing(output_dir) ?
        "$(MMM_DYNAMIC_DIR_PREFIX)_dyson_sign_Nw$(Nω)_Nk$(Nk)_dw$(dw_token)" :
        String(output_dir)
    mkpath(resolved_output_dir)

    Ω_values = dω .* Float64.(Ω_shifts)
    q_values = collect(2π .* (0:(Nk - 1)) ./ Nk)
    metadata = (
        Γs=collect(Float64.(Γs)),
        Nω=Nω,
        Nk=Nk,
        dω=dω,
        Ω_shifts=collect(Int.(Ω_shifts)),
        Ω_values=Ω_values,
        r=r,
        η=η,
    )
    jldsave(joinpath(resolved_output_dir, "metadata.jld2"); metadata...)

    ωs = physical_frequency_grid(Nω, dω)

    for Γraw in Γs
        Γ = Float64(Γraw)
        println("scanning --- Ω-line for Γ=", Γ)
        exact_GR, exact_GA, exact_GK = build_exact_mm_components(
            ωs,
            q_values;
            r=r,
            Γ=Γ,
            η=η,
            distribution_mode=:dyson_sign,
        )
        reduced_GR, reduced_GA, reduced_GK = build_reduced_dynamic_mm_components(
            ωs,
            q_values;
            r=r,
            Γ=Γ,
            η=η,
        )

        exact_values = compute_scalar_cycle_line(exact_GR, exact_GA, exact_GK, Ω_shifts, dω)
        reduced_values = compute_scalar_cycle_line(reduced_GR, reduced_GA, reduced_GK, Ω_shifts, dω)

        pos_sel = findall(>(0.0), Ω_values)
        exact_fit = fit_odd_omega_log_line(Ω_values[pos_sel], exact_values[pos_sel])
        reduced_fit = fit_odd_omega_log_line(Ω_values[pos_sel], reduced_values[pos_sel])

        first_pos = findfirst(>(0), Ω_shifts)
        qdiff = qresolved_cycle_difference(exact_GR, exact_GA, exact_GK, Ω_shifts[first_pos], dω)
        signed_q = q_to_signed.(q_values)
        peak_idx = sortperm(abs.(imag.(qdiff)); rev=true)[1:12]
        q_peaks = [(signed_q[i], qdiff[i]) for i in peak_idx]

        result = (
            Γ=Γ,
            Ω_values=Ω_values,
            exact_values=exact_values,
            reduced_values=reduced_values,
            exact_fit_coeff_log=exact_fit.coeff_log,
            exact_fit_coeff_linear=exact_fit.coeff_linear,
            exact_fit_rel_rms=exact_fit.rel_rms,
            reduced_fit_coeff_log=reduced_fit.coeff_log,
            reduced_fit_coeff_linear=reduced_fit.coeff_linear,
            reduced_fit_rel_rms=reduced_fit.rel_rms,
            signed_q=signed_q,
            qresolved_difference=qdiff,
        )
        jldsave(joinpath(resolved_output_dir, "gamma_$(gamma_token(Γ)).jld2"); result...)
        write_dynamic_report(
            resolved_output_dir,
            Γ,
            Ω_values,
            exact_values,
            reduced_values,
            exact_fit,
            reduced_fit,
            q_peaks,
        )
    end

    return resolved_output_dir
end

function main()
    output_dir = collect_cubic_mmm_dynamic_line()
    println("saved dynamic --- scan to ", output_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
