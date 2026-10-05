using Printf

include("estimate_edge_Aomegak.jl")
include("verify_generic_r_regular_terms.jl")

function find_index(values::AbstractVector{<:Real}, target::Real; atol::Float64=1e-15)
    idx = findfirst(v -> isapprox(v, target; atol=atol, rtol=0.0), values)
    idx === nothing && error("Target value $(target) not found in grid $(values)")
    return idx
end

analytic_smooth_background_delta(Gamma::Real, Omega::Real, k::Real) =
    -2.0 * (Float64(Omega)^2 + Float64(k)^2) / Float64(Gamma)^2

function analytic_smooth_background_log_shift(r::Real, Gamma::Real, Omega::Real, k::Real)
    p = generic_r_parameters(r)
    Lomega = 1.0 / (8.0 * pi * p.s)
    delta = Float64(Omega) - p.c * Float64(k)
    return -analytic_smooth_background_delta(Gamma, Omega, k) / (Lomega * delta^2)
end

function estimate_C_from_full_green_for_r(
    r::Real,
    Gamma::Real,
    omega_targets::AbstractVector{<:Real},
    k_targets::AbstractVector{<:Real};
    Nw::Int=8192,
    Nq::Int=2048,
    wmax::Float64=640.0,
    eta::Float64=1e-4,
)
    omega_values = sort(unique(vcat(0.0, Float64.(collect(omega_targets)))))
    k_values = sort(unique(vcat(0.0, Float64.(collect(k_targets)))))

    ws, dw = midpoint_frequency_grid(Nw, wmax)
    qs, _ = uniform_momentum_grid(Nq)
    comp = component_direct_mixed_grid_for_gamma(
        Float64(Gamma),
        ws,
        dw,
        qs,
        omega_values,
        k_values;
        r=Float64(r),
        η=Float64(eta),
    )

    iO0 = find_index(omega_values, 0.0)
    ik0 = find_index(k_values, 0.0)
    Pi00 = comp.full[iO0, ik0]

    p = generic_r_parameters(r)
    Lomega = 1.0 / (8.0 * pi * p.s)
    C_exact = endpoint_matching_constant(r)
    lnC_exact = log(C_exact)

    rows = NamedTuple[]
    for (Omega, k) in ((1e-4, 0.0), (2e-4, 7e-5))
        iO = find_index(omega_values, Omega)
        ik = find_index(k_values, k)
        Pi = comp.full[iO, ik]
        Delta_full = Float64(Gamma) * real(Pi - Pi00)
        Delta_reg = regular_delta(r, Omega, k)
        delta = Float64(Omega) - p.c * Float64(k)
        lnC_est = log(Float64(Gamma) * abs(delta)) - (Delta_full - Delta_reg) / (Lomega * delta^2)
        C_est = exp(lnC_est)
        lnC_smooth = analytic_smooth_background_log_shift(r, Gamma, Omega, k)
        push!(rows, (
            Omega=Float64(Omega),
            k=Float64(k),
            delta=delta,
            Delta_full=Delta_full,
            Delta_reg=Delta_reg,
            Delta_smooth_analytic=analytic_smooth_background_delta(Gamma, Omega, k),
            lnC_smooth_analytic=lnC_smooth,
            lnC_est=lnC_est,
            C_est=C_est,
        ))
    end

    return (r=Float64(r), Gamma=Float64(Gamma), C_exact=C_exact, lnC_exact=lnC_exact, rows=rows)
end

function main(; Nw::Int=8192, Nq::Int=2048, wmax::Float64=640.0, Gamma::Float64=80.0, eta::Float64=1e-4)
    rs = (0.5, 1.0, 1.5)
    println("Full-Green C_end estimator check")
    println(@sprintf("Nw=%d Nq=%d wmax=%.1f Gamma=%.1f eta=%.1e", Nw, Nq, wmax, Gamma, eta))
    println()
    println("r      C_exact       C_est(O=1e-4,k=0)    C_est(O=2e-4,k=7e-5)")

    results = NamedTuple[]
    for r in rs
        out = estimate_C_from_full_green_for_r(r, Gamma, [1e-4, 2e-4], [7e-5]; Nw=Nw, Nq=Nq, wmax=wmax, eta=eta)
        c1 = out.rows[1].C_est
        c2 = out.rows[2].C_est
        println(@sprintf("%3.1f  %11.4f    %16.4f    %19.4f", out.r, out.C_exact, c1, c2))
        push!(results, out)
    end

    println()
    println("r      ln C_exact    ln C_est(O=1e-4,k=0)  ln C_est(O=2e-4,k=7e-5)")
    for out in results
        l1 = out.rows[1].lnC_est
        l2 = out.rows[2].lnC_est
        println(@sprintf("%3.1f  %11.6f    %18.6f    %21.6f", out.r, out.lnC_exact, l1, l2))
    end

    println()
    println("r      analytic dln_smooth(1e-4,0)   analytic dln_smooth(2e-4,7e-5)")
    for out in results
        s1 = out.rows[1].lnC_smooth_analytic
        s2 = out.rows[2].lnC_smooth_analytic
        println(@sprintf("%3.1f  %24.6f   %27.6f", out.r, s1, s2))
    end

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
