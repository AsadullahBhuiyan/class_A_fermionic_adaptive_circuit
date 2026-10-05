using Printf
using Base.Threads: @threads, maxthreadid, threadid

include("estimate_edge_Aomegak.jl")
include("touching_model.jl")
include("verify_generic_r_regular_terms.jl")

const MODEL_ETA = 1e-4

function find_index(values::AbstractVector{<:Real}, target::Real; atol::Float64=1e-15)
    idx = findfirst(v -> isapprox(v, target; atol=atol, rtol=0.0), values)
    idx === nothing && error("Target value $(target) not found in grid $(values)")
    return idx
end

function edge_q_grid(r::Real, Nq::Int)
    p = generic_r_parameters(r)
    dq = 2.0 * p.qr / Nq
    qs = Vector{Float64}(undef, Nq)
    @inbounds for j in 1:Nq
        qs[j] = -p.qr + (j - 0.5) * dq
    end
    return qs, dq
end

@inline function edge_residue(r::Float64, q::Float64)
    mass = r - cos(q)
    return 1.0 - mass^2
end

@inline function gmm_local_old(omega::Float64, q::Float64, r::Float64, Gamma::Float64)
    z = edge_residue(r, q)
    z <= 0.0 && return 0.0 + 0.0im
    x = omega - sin(q)
    return x / (im * Gamma * x / 2.0 - z)
end

@inline function gmm_local_improved(omega::Float64, q::Float64, r::Float64, Gamma::Float64)
    mass = r - cos(q)
    z = 1.0 - mass^2
    z <= 0.0 && return 0.0 + 0.0im
    x = omega - sin(q)
    denom = -z + (2.0 * sin(q) / z) * x + im * Gamma * x / 2.0
    return x / denom
end

@inline function gmm_local_improved_schur(omega::Float64, q::Float64, r::Float64, Gamma::Float64)
    mass = r - cos(q)
    z = 1.0 - mass^2
    z <= 0.0 && return 0.0 + 0.0im
    x = omega - sin(q)
    denom = -z + (2.0 * sin(q) / z) * x - mass^2 * x / (x + im * Gamma / 2.0) + im * Gamma * x / 2.0
    return x / denom
end

@inline function gmm_uniform_lowerbranch(omega::Float64, q::Float64, r::Float64, Gamma::Float64)
    sq = sin(q)
    mass = r - cos(q)
    z = 1.0 - mass^2
    z <= 0.0 && return 0.0 + 0.0im
    x = ComplexF64(omega - sq, MODEL_ETA)
    lower_arg = x^2 + 2.0 * sq * x - (1.0 - mass)^2
    sigma_uniform = (x^2 + 2.0 * sq * x + z - im * (1.0 + mass) * sqrt(lower_arg)) / (2.0 * x)
    denom = x + 2.0 * sq - sigma_uniform + im * Gamma / 2.0 - mass^2 / (x + im * Gamma / 2.0)
    return inv(denom)
end

@inline function gmm_touching_local(omega::Float64, q::Float64, r::Float64, Gamma::Float64)
    @assert isapprox(r, 1.0; atol=1e-12, rtol=0.0) "touching model currently implemented only for r=1"
    _, _, _, gmm = local_touching_xbasis_entries(omega, q, Gamma, 1e-4)
    return gmm
end

@inline function gmm_crossover_frozen_smooth(omega::Float64, q::Float64, r::Float64, Gamma::Float64)
    sq = sin(q)
    mass = r - cos(q)
    zedge = 1.0 - mass^2
    zedge <= 0.0 && return 0.0 + 0.0im
    x = ComplexF64(omega - sq, MODEL_ETA)
    lower_arg = x^2 + 2.0 * sq * x - (1.0 - mass)^2
    v = sqrt(lower_arg)
    smooth0 = -sq * (1.0 - mass) / (2.0 * (1.0 + mass))
    denom = 0.5 * (im * (1.0 + mass) * v - zedge) + x * (sq - smooth0 + im * Gamma / 2.0)
    return x / denom
end

function scalar_model_bubble(
    model::Function,
    r::Float64,
    Gamma::Float64,
    omegas::Vector{Float64},
    domega::Float64,
    qs::Vector{Float64},
    dq::Float64,
    targets::AbstractVector,
)
    nt = maxthreadid()
    partial_base = zeros(ComplexF64, nt)
    partial_targets = zeros(ComplexF64, nt, length(targets))

    @threads for jq in eachindex(qs)
        tid = threadid()
        q = qs[jq]
        base_local = 0.0 + 0.0im
        target_local = zeros(ComplexF64, length(targets))

        @inbounds for omega in omegas
            g = model(omega, q, r, Gamma)
            base_local += g * conj(g)
            for it in eachindex(targets)
                tgt = targets[it]
                h = model(omega - tgt.Omega, q - tgt.k, r, Gamma)
                target_local[it] += g * conj(h)
            end
        end

        partial_base[tid] += base_local
        for it in eachindex(targets)
            partial_targets[tid, it] += target_local[it]
        end
    end

    prefactor = domega * dq / (2.0 * pi)^2
    return prefactor * sum(partial_base), prefactor .* vec(sum(partial_targets; dims=1))
end

function lnC_from_delta(r::Float64, Gamma::Float64, Omega::Float64, k::Float64, delta_val::Float64)
    p = generic_r_parameters(r)
    delta = Omega - p.c * k
    Lomega = 1.0 / (8.0 * pi * p.s)
    return log(Gamma * abs(delta)) - (delta_val - regular_delta(r, Omega, k)) / (Lomega * delta^2)
end

function compare_missing_piece_for_r(
    r::Real,
    Gamma::Real;
    Nw::Int=32768,
    Nq_full::Int=4096,
    Nq_edge::Int=8192,
    wmax::Float64=640.0,
    eta::Float64=1e-4,
)
    rf = Float64(r)
    Gf = Float64(Gamma)
    targets = [(Omega=1e-4, k=0.0), (Omega=2e-4, k=7e-5)]
    omega_values = sort(unique(vcat(0.0, [t.Omega for t in targets]...)))
    k_values = sort(unique(vcat(0.0, [t.k for t in targets]...)))

    omegas, domega = midpoint_frequency_grid(Nw, wmax)
    qs_full, _ = uniform_momentum_grid(Nq_full)
    exact = component_direct_mixed_grid_for_gamma(Gf, omegas, domega, qs_full, omega_values, k_values; r=rf, η=Float64(eta))

    iO0 = find_index(omega_values, 0.0)
    ik0 = find_index(k_values, 0.0)
    pi00_full = exact.full[iO0, ik0]
    pi00_mm = exact.mm[iO0, ik0]

    qs_edge, dq_edge = edge_q_grid(rf, Nq_edge)
    pi00_old, pits_old = scalar_model_bubble(gmm_local_old, rf, Gf, omegas, domega, qs_edge, dq_edge, targets)
    pi00_imp, pits_imp = scalar_model_bubble(gmm_local_improved, rf, Gf, omegas, domega, qs_edge, dq_edge, targets)
    pi00_imp_s, pits_imp_s = scalar_model_bubble(gmm_local_improved_schur, rf, Gf, omegas, domega, qs_edge, dq_edge, targets)

    rows = NamedTuple[]
    for (it, tgt) in enumerate(targets)
        iO = find_index(omega_values, tgt.Omega)
        ik = find_index(k_values, tgt.k)
        delta_full = Gf * real(exact.full[iO, ik] - pi00_full)
        delta_mm = Gf * real(exact.mm[iO, ik] - pi00_mm)
        delta_old = Gf * real(pits_old[it] - pi00_old)
        delta_imp = Gf * real(pits_imp[it] - pi00_imp)
        delta_imp_s = Gf * real(pits_imp_s[it] - pi00_imp_s)
        push!(rows, (
            Omega=tgt.Omega,
            k=tgt.k,
            lnC_full=lnC_from_delta(rf, Gf, tgt.Omega, tgt.k, delta_full),
            lnC_mm=lnC_from_delta(rf, Gf, tgt.Omega, tgt.k, delta_mm),
            lnC_old=lnC_from_delta(rf, Gf, tgt.Omega, tgt.k, delta_old),
            lnC_improved=lnC_from_delta(rf, Gf, tgt.Omega, tgt.k, delta_imp),
            lnC_improved_schur=lnC_from_delta(rf, Gf, tgt.Omega, tgt.k, delta_imp_s),
            lnC_uniform=NaN,
            lnC_touching=NaN,
            lnC_crossover=NaN,
        ))
    end

    pi00_uniform, pits_uniform = scalar_model_bubble(gmm_uniform_lowerbranch, rf, Gf, omegas, domega, qs_edge, dq_edge, targets)
    rows = NamedTuple[
        merge(row, (
            lnC_uniform=lnC_from_delta(rf, Gf, row.Omega, row.k, Gf * real(pits_uniform[i] - pi00_uniform)),
        )) for (i, row) in enumerate(rows)
    ]

    pi00_cross, pits_cross = scalar_model_bubble(gmm_crossover_frozen_smooth, rf, Gf, omegas, domega, qs_edge, dq_edge, targets)
    rows = NamedTuple[
        merge(row, (
            lnC_crossover=lnC_from_delta(rf, Gf, row.Omega, row.k, Gf * real(pits_cross[i] - pi00_cross)),
        )) for (i, row) in enumerate(rows)
    ]

    if isapprox(rf, 1.0; atol=1e-12, rtol=0.0)
        pi00_touch, pits_touch = scalar_model_bubble(gmm_touching_local, rf, Gf, omegas, domega, qs_full, 2.0 * pi / Nq_full, targets)
        rows = NamedTuple[
            merge(row, (
                lnC_touching=lnC_from_delta(rf, Gf, row.Omega, row.k, Gf * real(pits_touch[i] - pi00_touch)),
            )) for (i, row) in enumerate(rows)
        ]
    end

    return (r=rf, Gamma=Gf, lnC_exact=log(endpoint_matching_constant(rf)), rows=rows)
end

function main(; Nw::Int=32768, Nq_full::Int=4096, Nq_edge::Int=8192, wmax::Float64=640.0, Gamma::Float64=80.0, eta::Float64=1e-4)
    println("Missing-piece test in the -- channel")
    println(@sprintf("Nw=%d Nq_full=%d Nq_edge=%d wmax=%.1f Gamma=%.1f eta=%.1e", Nw, Nq_full, Nq_edge, wmax, Gamma, eta))
    println()

    results = [compare_missing_piece_for_r(r, Gamma; Nw=Nw, Nq_full=Nq_full, Nq_edge=Nq_edge, wmax=wmax, eta=eta) for r in (0.5, 1.0, 1.5)]
    for out in results
        println(@sprintf("r=%.1f lnC_exact=%.6f", out.r, out.lnC_exact))
        for row in out.rows
            println(
                @sprintf(
                    "  (Omega=%.1e, k=%.1e): full=%.6f mm=%.6f old=%.6f improved=%.6f improved+schur=%.6f uniform=%.6f crossing=%.6f touching=%.6f",
                    row.Omega,
                    row.k,
                    row.lnC_full,
                    row.lnC_mm,
                    row.lnC_old,
                    row.lnC_improved,
                    row.lnC_improved_schur,
                    row.lnC_uniform,
                    row.lnC_crossover,
                    row.lnC_touching,
                ),
            )
        end
        println()
    end

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
