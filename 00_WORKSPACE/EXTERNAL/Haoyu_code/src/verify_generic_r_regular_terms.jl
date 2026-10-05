using Printf

function generic_r_parameters(r::Real)
    @assert 0 < r < 2
    rf = Float64(r)
    c = rf - 1.0
    s = sqrt(rf * (2.0 - rf))
    qr = acos(c)
    J = 4.0 * acos(rf / 2.0) / sqrt(rf * (rf + 2.0))
    Areg = (rf * J - qr) / (4.0 * pi)
    Breg = J / (4.0 * pi)
    return (r=rf, c=c, s=s, qr=qr, J=J, Areg=Areg, Breg=Breg)
end

function endpoint_matching_constant(r::Real)
    p = generic_r_parameters(r)
    return 16.0 * p.s^2 * exp(0.5 * p.s * p.J)
end

function residue_integral(r::Real)
    p = generic_r_parameters(r)
    return (1.0 - 2.0 * p.r^2) * p.qr + (3.0 * p.r + 1.0) * p.s
end

function large_gamma_endpoint_asymptote(r::Real, Gamma::Real)
    return -2.0 * residue_integral(r) / (pi * Float64(Gamma)^2)
end

@inline residue_z(r::Float64, q::Float64) = 1.0 - (r - cos(q))^2

function endpoint_delta(r::Real, Gamma::Real, Omega::Real, k::Real; Nq::Int=1_000_000)
    p = generic_r_parameters(r)
    rf = p.r
    qr = p.qr
    dq = 2.0 * qr / Nq
    acc = 0.0
    @inbounds for j in 1:Nq
        q = -qr + (j - 0.5) * dq
        z = residue_z(rf, q)
        d = Float64(Omega) - Float64(k) * cos(q)
        acc += -z * d^2 / (4.0 * z^2 + Float64(Gamma)^2 * d^2 / 4.0)
    end
    return acc * dq / (2.0 * pi)
end

function singular_log_term(r::Real, Gamma::Real, Omega::Real, k::Real)
    p = generic_r_parameters(r)
    d = Float64(Omega) - p.c * Float64(k)
    if iszero(d)
        return 0.0
    end
    return -(d^2) / (8.0 * pi * p.s) * log(1.0 / (Float64(Gamma) * abs(d)))
end

function regular_delta(r::Real, Omega::Real, k::Real)
    p = generic_r_parameters(r)
    return p.Breg * Float64(Omega) * Float64(k) - p.Areg * Float64(k)^2
end

function matching_constant_estimate(
    r::Real,
    Gamma::Real,
    Omega::Real,
    k::Real;
    Nq::Int=1_000_000,
)
    p = generic_r_parameters(r)
    delta = Float64(Omega) - p.c * Float64(k)
    @assert !iszero(delta)
    Lomega = 1.0 / (8.0 * pi * p.s)
    delta_end = endpoint_delta(r, Gamma, Omega, k; Nq=Nq)
    delta_reg = regular_delta(r, Omega, k)
    return Float64(Gamma) * abs(delta) *
           exp(-(delta_end - delta_reg) / (Lomega * delta^2))
end

fit_quadratic(x::AbstractVector, y::AbstractVector) =
    sum((x .^ 2) .* y) / sum(x .^ 4)

function odd_odd_2x2(values::AbstractMatrix{<:Real})
    @assert size(values) == (2, 2)
    return 0.25 * (values[2, 2] - values[2, 1] - values[1, 2] + values[1, 1])
end

function verify_regular_terms(r::Real, Gamma::Real; Nq_line::Int=1_000_000, Nq_mixed::Int=500_000)
    p = generic_r_parameters(r)
    kvals = [5e-4, 1e-3, 1.5e-3, 2e-3, 2.5e-3]
    Ovals = [5e-4, 1e-3, 1.5e-3]

    yO = [endpoint_delta(r, Gamma, O, 0.0; Nq=Nq_line) for O in Ovals]
    yk = [endpoint_delta(r, Gamma, 0.0, k; Nq=Nq_line) for k in kvals]

    chi = fit_quadratic(Ovals, yO .- [singular_log_term(r, Gamma, O, 0.0) for O in Ovals])
    Rk = fit_quadratic(kvals, yk .- [singular_log_term(r, Gamma, 0.0, k) for k in kvals])
    Areg_num = chi * p.c^2 - Rk

    Opos = [1e-3]
    kpos = [1e-3]
    Os = [-Opos[1], Opos[1]]
    ks = [-kpos[1], kpos[1]]
    vals = [endpoint_delta(r, Gamma, O, k; Nq=Nq_mixed) for O in Os, k in ks]
    logs = [singular_log_term(r, Gamma, O, k) for O in Os, k in ks]
    Rodd = odd_odd_2x2(vals .- logs) / (Opos[1] * kpos[1])
    Breg_num = Rodd + 2.0 * chi * p.c

    return (
        r=p.r,
        Gamma=Float64(Gamma),
        Areg_num=Areg_num,
        Areg_exact=p.Areg,
        Breg_num=Breg_num,
        Breg_exact=p.Breg,
        chi=chi,
    )
end

function main(; r_values=(0.5, 1.0, 1.5), Gamma::Real=80.0)
    println("Reduced endpoint-kernel checks")
    println("Gamma = ", Float64(Gamma))
    println()
    println("Regular-term verification")
    println("r      Areg_num    Areg_exact  Breg_num    Breg_exact  chi")
    for r in r_values
        row = verify_regular_terms(r, Gamma)
        println(
            @sprintf(
                "%4.1f  %10.6f  %10.6f  %10.6f  %10.6f  % .6f",
                row.r,
                row.Areg_num,
                row.Areg_exact,
                row.Breg_num,
                row.Breg_exact,
                row.chi,
            ),
        )
    end

    println()
    println("Endpoint matching-constant verification")
    println("r      C_exact     C_est(1e-4,0)  C_est(2e-4,7e-5)")
    for r in r_values
        C_exact = endpoint_matching_constant(r)
        C_line = matching_constant_estimate(r, Gamma, 1e-4, 0.0)
        C_mixed = matching_constant_estimate(r, Gamma, 2e-4, 7e-5)
        println(
            @sprintf(
                "%4.1f  %10.4f  %14.4f  %16.4f",
                Float64(r),
                C_exact,
                C_line,
                C_mixed,
            ),
        )
    end

    Gamma_large = 2000.0
    k_large = 0.02
    Omega_off = 0.03
    println()
    println("Large-Gamma*Q saturation check")
    println("Gamma = ", Gamma_large, ", k = ", k_large)
    println("off-tangent point: Omega = ", Omega_off, ", k = ", k_large)
    println("r      Delta_sat       tangent_ratio  off_tangent_ratio")
    for r in r_values
        p = generic_r_parameters(r)
        Delta_sat = large_gamma_endpoint_asymptote(r, Gamma_large)
        Delta_tan = endpoint_delta(r, Gamma_large, p.c * k_large, k_large)
        Delta_off = endpoint_delta(r, Gamma_large, Omega_off, k_large)
        println(
            @sprintf(
                "%4.1f  % .6e   %12.6f  %16.6f",
                Float64(r),
                Delta_sat,
                Delta_tan / Delta_sat,
                Delta_off / Delta_sat,
            ),
        )
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
