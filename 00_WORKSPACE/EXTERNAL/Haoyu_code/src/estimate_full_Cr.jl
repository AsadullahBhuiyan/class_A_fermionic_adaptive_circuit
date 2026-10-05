using JLD2
using Printf
using Base.Threads: @threads, maxthreadid, threadid

include("verify_full_large_gammaQ.jl")

@inline function xbasis_mm_entry(
    g11::ComplexF64,
    g12::ComplexF64,
    g21::ComplexF64,
    g22::ComplexF64,
)
    return 0.5 * (g11 - g12 - g21 + g22)
end

function signed_values_from_positive(pos::AbstractVector{<:Real})
    vals = Float64.(pos)
    return vcat(-reverse(vals), vals)
end

function append_midpoint_segment!(
    xs::Vector{Float64},
    ws::Vector{Float64},
    a::Float64,
    b::Float64,
    step::Float64,
)
    if b <= a
        return nothing
    end
    n = max(1, ceil(Int, (b - a) / step))
    h = (b - a) / n
    @inbounds for i in 1:n
        push!(xs, a + (i - 0.5) * h)
        push!(ws, h)
    end
    return nothing
end

function composite_momentum_grid(
    r::Real;
    endpoint_window::Real=0.08,
    dense_step::Real=2.0e-4,
    coarse_step::Real=3.0e-3,
)
    p = generic_r_parameters(r)
    qr = p.qr
    w = min(Float64(endpoint_window), 0.45 * qr, 0.45 * (pi - qr))
    breaks = [
        -pi,
        -qr - w,
        -qr + w,
        qr - w,
        qr + w,
        pi,
    ]

    xs = Float64[]
    ws = Float64[]
    for i in 1:(length(breaks)-1)
        a = breaks[i]
        b = breaks[i+1]
        mid = 0.5 * (a + b)
        near_endpoint = abs(mid + qr) < w || abs(mid - qr) < w
        step = near_endpoint ? Float64(dense_step) : Float64(coarse_step)
        append_midpoint_segment!(xs, ws, a, b, step)
    end
    return xs, ws
end

function symmetrized_positive(values::AbstractVector{ComplexF64})
    n2 = length(values)
    @assert iseven(n2)
    n = n2 ÷ 2
    out = Vector{ComplexF64}(undef, n)
    @inbounds for i in 1:n
        out[i] = 0.5 * (values[n + i] + values[n - i + 1])
    end
    return out
end

function full_and_mm_omega_lines_for_gamma(
    Gamma::Real,
    omegas::Vector{Float64},
    omega_weights::Vector{Float64},
    qs::Vector{Float64},
    q_weights::Vector{Float64},
    Omega_values_signed::Vector{Float64};
    r::Real=1.0,
    eta::Real=1e-6,
)
    @assert length(omegas) == length(omega_weights)
    @assert length(qs) == length(q_weights)
    G = Float64(Gamma)
    rr = Float64(r)
    et = Float64(eta)
    nOmega = length(Omega_values_signed)
    nt = maxthreadid()

    partial_base_full = zeros(ComplexF64, nt)
    partial_base_mm = zeros(ComplexF64, nt)
    partial_full = zeros(ComplexF64, nt, nOmega)
    partial_mm = zeros(ComplexF64, nt, nOmega)

    @threads for jq in eachindex(qs)
        tid = threadid()
        q = qs[jq]
        q_weight = q_weights[jq]
        base_full_local = 0.0 + 0.0im
        base_mm_local = 0.0 + 0.0im

        @inbounds for iomega in eachindex(omegas)
            omega = omegas[iomega]
            weight = omega_weights[iomega] * q_weight
            g11, g12, g21, g22 = monitored_retarded_green_entries(omega, q, rr, G, et)
            gmm = xbasis_mm_entry(g11, g12, g21, g22)

            base_full_local += weight * overlap_entries(g11, g12, g21, g22, g11, g12, g21, g22)
            base_mm_local += weight * gmm * conj(gmm)

            for iOmega in eachindex(Omega_values_signed)
                Omega = Omega_values_signed[iOmega]
                h11, h12, h21, h22 = monitored_retarded_green_entries(omega - Omega, q, rr, G, et)
                hmm = xbasis_mm_entry(h11, h12, h21, h22)
                partial_full[tid, iOmega] += weight * overlap_entries(g11, g12, g21, g22, h11, h12, h21, h22)
                partial_mm[tid, iOmega] += weight * gmm * conj(hmm)
            end
        end

        partial_base_full[tid] += base_full_local
        partial_base_mm[tid] += base_mm_local
    end

    prefactor = 1.0 / (4π^2)
    return (
        Pi00_full=prefactor * sum(partial_base_full),
        Pi00_mm=prefactor * sum(partial_base_mm),
        Pi_full=prefactor .* vec(sum(partial_full; dims=1)),
        Pi_mm=prefactor .* vec(sum(partial_mm; dims=1)),
    )
end

function estimate_lnC_rows(
    r::Real,
    Gamma::Real,
    Omega_values::Vector{Float64},
    line_data,
)
    p = generic_r_parameters(r)
    Lomega = 1.0 / (8.0 * pi * p.s)
    Pi_full_sym = symmetrized_positive(line_data.Pi_full)
    Pi_mm_sym = symmetrized_positive(line_data.Pi_mm)
    rows = Vector{NamedTuple}(undef, length(Omega_values))

    @inbounds for i in eachindex(Omega_values)
        Omega = Omega_values[i]
        Delta_full = real(Gamma * (Pi_full_sym[i] - line_data.Pi00_full))
        Delta_mm = real(Gamma * (Pi_mm_sym[i] - line_data.Pi00_mm))
        lnC_full = log(Gamma * abs(Omega)) - Delta_full / (Lomega * Omega^2)
        lnC_mm = log(Gamma * abs(Omega)) - Delta_mm / (Lomega * Omega^2)
        rows[i] = (
            r=Float64(r),
            Gamma=Float64(Gamma),
            Omega=Omega,
            Delta_full=Delta_full,
            Delta_mm=Delta_mm,
            lnC_full=lnC_full,
            lnC_mm=lnC_mm,
            lnC_end=log(endpoint_matching_constant(r)),
        )
    end
    return rows
end

function main(;
    r_values=(0.5, 1.0, 1.5),
    Gamma::Real=80.0,
    Omega_values=(1.5e-3,),
    frequency_step_factor::Real=100.0,
    wmax::Real=640.0,
    eta::Real=1e-6,
    output_path::AbstractString="data/full_green_lnCr_refined.jld2",
)
    mkpath(dirname(output_path))
    Opos = collect(Float64.(Omega_values))
    Osigned = signed_values_from_positive(Opos)
    freq_step = 1.0 / (Float64(frequency_step_factor) * Float64(Gamma))
    omegas, omega_weights = composite_frequency_grid(Gamma; wmax=wmax, dense_step=freq_step)

    all_rows = NamedTuple[]
    raw = Dict{String,Any}()

    println("Full Green-function ln(C_r) extraction")
    println(
        "Gamma = ",
        Float64(Gamma),
        ", Nomega = ",
        length(omegas),
        ", dense frequency step = ",
        freq_step,
    )
    println("Omega values = ", Opos)
    println()
    println(" r     Omega      lnC_full    lnC_mm      lnC_end    Delta_full    Delta_mm")

    for r in r_values
        qs, q_weights = composite_momentum_grid(r)
        println("# r = ", Float64(r), ", Nq = ", length(qs))
        line = full_and_mm_omega_lines_for_gamma(
            Gamma,
            omegas,
            omega_weights,
            qs,
            q_weights,
            Osigned;
            r=r,
            eta=eta,
        )
        rows = estimate_lnC_rows(r, Float64(Gamma), Opos, line)
        append!(all_rows, rows)

        key = @sprintf("r_%0.3f", Float64(r))
        raw[key] = Dict(
            "Pi00_full" => line.Pi00_full,
            "Pi00_mm" => line.Pi00_mm,
            "Pi_full" => line.Pi_full,
            "Pi_mm" => line.Pi_mm,
            "q_grid" => qs,
            "q_weights" => q_weights,
        )

        for row in rows
            println(
                @sprintf(
                    "%3.1f  %9.2e  %10.5f  %10.5f  %10.5f  % .6e  % .6e",
                    row.r,
                    row.Omega,
                    row.lnC_full,
                    row.lnC_mm,
                    row.lnC_end,
                    row.Delta_full,
                    row.Delta_mm,
                ),
            )
        end
    end

    jldsave(
        output_path;
        rows=all_rows,
        r_values=collect(Float64.(r_values)),
        Gamma=Float64(Gamma),
        Omega_values=Opos,
        Omega_values_signed=Osigned,
        wmax=Float64(wmax),
        eta=Float64(eta),
        Nomega=length(omegas),
        frequency_step_factor=Float64(frequency_step_factor),
        dense_frequency_step=freq_step,
        omega_grid=omegas,
        omega_weights=omega_weights,
        raw=raw,
    )
    println()
    println("saved ", output_path)
    return all_rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
