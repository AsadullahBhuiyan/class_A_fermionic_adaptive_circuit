using Printf
using Base.Threads: @threads, maxthreadid, threadid

include("collect_small_argument_direct_lines.jl")
include("verify_generic_r_regular_terms.jl")

function full_large_gamma_asymptote(r::Real, Gamma::Real, Omega::Real, k::Real)
    G = Float64(Gamma)
    O = Float64(Omega)
    kk = Float64(k)
    return 2im * O / G - (2.0 * residue_integral(r) / pi + 2.0 * O^2 + 2.0 * kk^2) / G^2
end

function midpoint_segment(a::Real, b::Real, n::Integer)
    @assert b > a
    @assert n > 0
    af = Float64(a)
    bf = Float64(b)
    nf = Int(n)
    h = (bf - af) / nf
    xs = Vector{Float64}(undef, nf)
    ws = fill(h, nf)
    @inbounds for i in 1:nf
        xs[i] = af + (i - 0.5) * h
    end
    return xs, ws
end

function composite_frequency_grid(
    Gamma::Real;
    wlow::Real=2.0,
    wmax::Real=8.0 * Float64(Gamma),
    dense_step::Real=1.0 / (20.0 * Float64(Gamma)),
    coarse_step::Real=0.5,
)
    G = Float64(Gamma)
    wl = Float64(wlow)
    wm = Float64(wmax)
    @assert wm > wl

    nlow = ceil(Int, 2.0 * wl / Float64(dense_step))
    nleft = ceil(Int, (wm - wl) / Float64(coarse_step))
    nright = nleft

    xleft, wleft = midpoint_segment(-wm, -wl, nleft)
    xmid, wmid = midpoint_segment(-wl, wl, nlow)
    xright, wright = midpoint_segment(wl, wm, nright)

    return vcat(xleft, xmid, xright), vcat(wleft, wmid, wright)
end

function direct_full_point_for_gamma(
    Gamma::Real,
    omegas::Vector{Float64},
    omega_weights::Vector{Float64},
    qs::Vector{Float64},
    Omega::Real,
    k::Real;
    r::Real=1.0,
    eta::Real=1e-4,
)
    @assert length(omegas) == length(omega_weights)
    G = Float64(Gamma)
    O = Float64(Omega)
    kk = Float64(k)
    rr = Float64(r)
    et = Float64(eta)

    nt = maxthreadid()
    partial_base = zeros(ComplexF64, nt)
    partial_shifted = zeros(ComplexF64, nt)

    @threads for jq in eachindex(qs)
        tid = threadid()
        q = qs[jq]
        base_local = 0.0 + 0.0im
        shifted_local = 0.0 + 0.0im

        @inbounds for iω in eachindex(omegas)
            omega = omegas[iω]
            weight = omega_weights[iω]
            g11, g12, g21, g22 = monitored_retarded_green_entries(omega, q, rr, G, et)
            h11, h12, h21, h22 = monitored_retarded_green_entries(omega - O, q - kk, rr, G, et)
            base_local += weight * overlap_entries(g11, g12, g21, g22, g11, g12, g21, g22)
            shifted_local += weight * overlap_entries(g11, g12, g21, g22, h11, h12, h21, h22)
        end

        partial_base[tid] += base_local
        partial_shifted[tid] += shifted_local
    end

    prefactor = 1.0 / (2π * length(qs))
    Pi00 = prefactor * sum(partial_base)
    Pi_shifted = prefactor * sum(partial_shifted)
    return G * (Pi_shifted - Pi00)
end

function print_case(row)
    omegas, weights = composite_frequency_grid(row.Gamma; wmax=row.wmax)
    qs, _ = uniform_momentum_grid(row.Nq)
    num = direct_full_point_for_gamma(
        row.Gamma,
        omegas,
        weights,
        qs,
        row.Omega,
        row.k;
        r=row.r,
        eta=row.eta,
    )
    pred = full_large_gamma_asymptote(row.r, row.Gamma, row.Omega, row.k)
    println(
        @sprintf(
            "%3.1f %5.0f %8.3f %8.3f  % .6e  % .6e  % .6e  % .6e",
            row.r,
            row.Gamma,
            row.Omega,
            row.k,
            real(num),
            real(pred),
            imag(num),
            imag(pred),
        ),
    )
    return (num=num, pred=pred)
end

function main()
    println("Full Green-function large-Gamma*Q check")
    println("K = Gamma * (Pi_RA(Omega,k) - Pi_RA(0,0))")
    println("asymptote = 2 i Omega/Gamma - [2 I_Z(r)/pi + 2 Omega^2 + 2 k^2]/Gamma^2")
    println()
    println(" r  Gamma    Omega        k       Re K_num     Re K_asym     Im K_num     Im K_asym")

    rows = [
        (r=0.5, Gamma=200.0, Omega=0.1, k=0.0, Nq=512, wmax=1600.0, eta=1e-4),
        (r=1.0, Gamma=200.0, Omega=0.1, k=0.0, Nq=512, wmax=1600.0, eta=1e-4),
        (r=1.5, Gamma=200.0, Omega=0.1, k=0.0, Nq=512, wmax=1600.0, eta=1e-4),
        (r=1.0, Gamma=200.0, Omega=0.0, k=0.1, Nq=512, wmax=1600.0, eta=1e-4),
        (r=1.0, Gamma=400.0, Omega=0.0, k=0.1, Nq=512, wmax=3200.0, eta=1e-4),
        (r=1.5, Gamma=200.0, Omega=0.1, k=0.05, Nq=512, wmax=1600.0, eta=1e-4),
    ]

    for row in rows
        print_case(row)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
