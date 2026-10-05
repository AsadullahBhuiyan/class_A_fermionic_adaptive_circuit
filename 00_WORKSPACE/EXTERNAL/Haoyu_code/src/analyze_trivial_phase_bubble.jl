using JLD2
using Printf
using Base.Threads: @threads, maxthreadid, nthreads, threadid

include("taylor_fit_convolution.jl")

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
    wlow::Real=8.0,
    wmax::Real=max(320.0, 8.0 * Float64(Gamma)),
    dense_step::Real=0.025,
    coarse_step::Real=0.75,
)
    wl = Float64(wlow)
    wm = Float64(wmax)
    @assert wm > wl
    nmid = ceil(Int, 2.0 * wl / Float64(dense_step))
    nside = ceil(Int, (wm - wl) / Float64(coarse_step))
    left, wleft = midpoint_segment(-wm, -wl, nside)
    mid, wmid = midpoint_segment(-wl, wl, nmid)
    right, wright = midpoint_segment(wl, wm, nside)
    return vcat(left, mid, right), vcat(wleft, wmid, wright)
end

function uniform_momentum_grid(Nq::Integer)
    nq = Int(Nq)
    dq = 2π / nq
    qs = Vector{Float64}(undef, nq)
    weights = fill(dq, nq)
    @inbounds for j in 1:nq
        qs[j] = -π + (j - 0.5) * dq
    end
    return qs, weights
end

function signed_values_from_positive(pos)
    values = sort(unique(abs.(Float64.(collect(pos)))))
    return vcat(-reverse(values), values)
end

function target_shifts(omega_line_pos, k_line_pos)
    omega_signed = signed_values_from_positive(omega_line_pos)
    k_signed = signed_values_from_positive(k_line_pos)
    targets = Set{Tuple{Float64,Float64}}()
    push!(targets, (0.0, 0.0))
    for O in omega_signed
        push!(targets, (O, 0.0))
    end
    for k in k_signed
        push!(targets, (0.0, k))
    end
    for O in omega_signed, k in k_signed
        push!(targets, (O, k))
    end
    return sort(collect(targets); by=t -> (t[1], t[2]))
end

@inline function bubble_overlap(
    g11::ComplexF64,
    g12::ComplexF64,
    g21::ComplexF64,
    g22::ComplexF64,
    h11::ComplexF64,
    h12::ComplexF64,
    h21::ComplexF64,
    h22::ComplexF64,
)
    return g11 * conj(h11) + g12 * conj(h12) + g21 * conj(h21) + g22 * conj(h22)
end

@inline tail_correction_pi00(wmax::Real) = 2.0 / (π * Float64(wmax))

function bubble_targets_for_gamma(
    Gamma::Real,
    omegas::Vector{Float64},
    omega_weights::Vector{Float64},
    qs::Vector{Float64},
    q_weights::Vector{Float64},
    targets::Vector{Tuple{Float64,Float64}};
    r::Real=3.0,
    eta::Real=1e-6,
)
    @assert length(omegas) == length(omega_weights)
    @assert length(qs) == length(q_weights)
    G = Float64(Gamma)
    rr = Float64(r)
    et = Float64(eta)
    nt = maxthreadid()
    ntarget = length(targets)
    partial = zeros(ComplexF64, nt, ntarget)

    @threads for jq in eachindex(qs)
        tid = threadid()
        q = qs[jq]
        qw = q_weights[jq]
        local_acc = zeros(ComplexF64, ntarget)
        @inbounds for iω in eachindex(omegas)
            omega = omegas[iω]
            ow = omega_weights[iω]
            weight = ow * qw
            g11, g12, g21, g22 = monitored_retarded_green_entries(omega, q, rr, G, et)
            for it in eachindex(targets)
                O, k = targets[it]
                h11, h12, h21, h22 = monitored_retarded_green_entries(omega - O, q - k, rr, G, et)
                local_acc[it] += weight * bubble_overlap(g11, g12, g21, g22, h11, h12, h21, h22)
            end
        end
        @inbounds for it in eachindex(targets)
            partial[tid, it] += local_acc[it]
        end
    end

    prefactor = 1.0 / (2π)^2
    return prefactor .* vec(sum(partial; dims=1))
end

function mass_leakage_for_gamma(
    Gamma::Real,
    omegas::Vector{Float64},
    omega_weights::Vector{Float64},
    qs::Vector{Float64},
    q_weights::Vector{Float64};
    r::Real=3.0,
    eta::Real=1e-6,
)
    G = Float64(Gamma)
    rr = Float64(r)
    et = Float64(eta)
    nt = maxthreadid()
    partial = zeros(Float64, nt)

    @threads for jq in eachindex(qs)
        tid = threadid()
        q = qs[jq]
        qw = q_weights[jq]
        acc = 0.0
        @inbounds for iω in eachindex(omegas)
            omega = omegas[iω]
            sigma = Σbulk(ComplexF64(omega, et), q, rr)
            leak = -2.0 * imag(sigma)
            leak <= 0.0 && continue
            g11, g12, g21, g22 = monitored_retarded_green_entries(omega, q, rr, G, et)
            tr_gpmga = 0.5 * (abs2(g11 - g12) + abs2(g21 - g22))
            acc += omega_weights[iω] * qw * leak * tr_gpmga
        end
        partial[tid] += acc
    end

    return sum(partial) / (2π)^2
end

function complex_fit(xcols::Vector{Vector{Float64}}, y::Vector{ComplexF64})
    X = hcat(xcols...)
    coeffs = X \ y
    fitted = X * coeffs
    residuals = y .- fitted
    rel_rms = sqrt(sum(abs2, residuals) / length(y)) / max(maximum(abs.(y)), eps())
    return (coeffs=coeffs, fitted=fitted, residuals=residuals, rel_rms=rel_rms)
end

function fit_trivial_coefficients(
    Gamma::Real,
    targets::Vector{Tuple{Float64,Float64}},
    Pi::Vector{ComplexF64};
    wmax::Real,
    mass_leakage::Real,
)
    G = Float64(Gamma)
    zero_idx = findfirst(==((0.0, 0.0)), targets)
    @assert !isnothing(zero_idx)
    Pi00 = Pi[zero_idx]
    K0_raw = G * Pi00
    K0_tail = G * (Pi00 + tail_correction_pi00(wmax))

    omega_x = Float64[]
    omega_y = ComplexF64[]
    k_x = Float64[]
    k_y = ComplexF64[]
    mixed_x = Float64[]
    mixed_y = ComplexF64[]
    lookup = Dict(targets .=> Pi)

    for (O, k) in targets
        if k == 0.0 && O != 0.0
            push!(omega_x, O)
            push!(omega_y, G * (lookup[(O, 0.0)] - Pi00))
        elseif O == 0.0 && k != 0.0
            push!(k_x, k)
            push!(k_y, G * (lookup[(0.0, k)] - Pi00))
        elseif O != 0.0 && k != 0.0
            push!(mixed_x, O * k)
            push!(
                mixed_y,
                G * (lookup[(O, k)] - lookup[(O, 0.0)] - lookup[(0.0, k)] + Pi00),
            )
        end
    end

    omega_fit = complex_fit([omega_x, omega_x .^ 2], omega_y)
    k_fit = complex_fit([k_x, k_x .^ 2], k_y)
    mixed_fit = complex_fit([mixed_x], mixed_y)

    analytic = (
        K0=2.0 - 4.0 / G^2,
        mass_residual=4.0 / G^2,
        Lomega=2.0im / G,
        Lk=0.0 + 0.0im,
        Qomegaomega=-2.0 / G^2 + 0.0im,
        Qkk=-2.0 / G^2 + 0.0im,
        Qomegak=0.0 + 0.0im,
    )

    return (
        Gamma=G,
        Pi00=Pi00,
        K0_raw=K0_raw,
        K0_tail_corrected=K0_tail,
        mass_residual=2.0 - real(K0_tail),
        mass_leakage=Float64(mass_leakage),
        Lomega=omega_fit.coeffs[1],
        Qomegaomega=omega_fit.coeffs[2],
        Lk=k_fit.coeffs[1],
        Qkk=k_fit.coeffs[2],
        Qomegak=mixed_fit.coeffs[1],
        omega_rel_rms=omega_fit.rel_rms,
        k_rel_rms=k_fit.rel_rms,
        mixed_rel_rms=mixed_fit.rel_rms,
        analytic=analytic,
    )
end

function main(;
    r::Real=3.0,
    Gamma_values=(20.0, 40.0, 80.0, 160.0),
    omega_line_pos=(0.01, 0.02),
    k_line_pos=(0.01, 0.02),
    Nq::Integer=768,
    wlow::Real=8.0,
    dense_step::Real=0.025,
    coarse_step::Real=0.75,
    wmax_factor::Real=8.0,
    min_wmax::Real=320.0,
    eta::Real=1e-6,
    output_path::AbstractString="data/trivial_r3_bubble_scan.jld2",
)
    rr = Float64(r)
    targets = target_shifts(omega_line_pos, k_line_pos)
    qs, q_weights = uniform_momentum_grid(Nq)
    rows = NamedTuple[]
    raw = Dict{Float64,Any}()

    println("Trivial-phase RA bubble scan")
    println("r = ", rr, ", targets = ", length(targets), ", Nq = ", Nq)
    println("Expansion: Gamma*Pi_RA = C0 + Lomega*Omega + Lk*k + Qoo*Omega^2 + Qok*Omega*k + Qkk*k^2 + ...")
    println()
    println(" Gamma   Re C0       2-ReC0      m_leak      Im Lomega   Re Qoo      Im Lk       Re Qkk      Re Qok")

    for Graw in Gamma_values
        G = Float64(Graw)
        wm = max(Float64(min_wmax), Float64(wmax_factor) * G)
        omegas, omega_weights = composite_frequency_grid(
            G;
            wlow=wlow,
            wmax=wm,
            dense_step=dense_step,
            coarse_step=coarse_step,
        )
        Pi = bubble_targets_for_gamma(
            G,
            omegas,
            omega_weights,
            qs,
            q_weights,
            targets;
            r=rr,
            eta=eta,
        )
        mleak = mass_leakage_for_gamma(
            G,
            omegas,
            omega_weights,
            qs,
            q_weights;
            r=rr,
            eta=eta,
        )
        fit = fit_trivial_coefficients(G, targets, Pi; wmax=wm, mass_leakage=mleak)
        push!(rows, fit)
        raw[G] = (
            wmax=wm,
            omega_grid=omegas,
            omega_weights=omega_weights,
            targets=targets,
            Pi=Pi,
        )
        println(
            @sprintf(
                "%6.1f  % .8f  % .4e  % .4e  % .6e  % .6e  % .6e  % .6e  % .6e",
                G,
                real(fit.K0_tail_corrected),
                fit.mass_residual,
                fit.mass_leakage,
                imag(fit.Lomega),
                real(fit.Qomegaomega),
                imag(fit.Lk),
                real(fit.Qkk),
                real(fit.Qomegak),
            ),
        )
    end

    mkpath(dirname(output_path))
    jldsave(
        output_path;
        r=rr,
        Gamma_values=Float64.(collect(Gamma_values)),
        omega_line_pos=Float64.(collect(omega_line_pos)),
        k_line_pos=Float64.(collect(k_line_pos)),
        targets=targets,
        Nq=Int(Nq),
        q_grid=qs,
        q_weights=q_weights,
        wlow=Float64(wlow),
        dense_step=Float64(dense_step),
        coarse_step=Float64(coarse_step),
        wmax_factor=Float64(wmax_factor),
        min_wmax=Float64(min_wmax),
        eta=Float64(eta),
        rows=rows,
        raw=raw,
        nthreads=nthreads(),
    )
    println()
    println("saved ", output_path)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
