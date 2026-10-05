using JLD2
using Printf

include("estimate_full_Cr.jl")

function main(;
    r::Real=1.0,
    Gamma::Real=20.0,
    Omega_values=(5e-4, 1e-3, 2e-3, 4e-3, 8e-3),
    frequency_step_factor::Real=1000.0,
    q_dense_step::Real=1e-3,
    q_coarse_step::Real=6e-3,
    endpoint_window::Real=0.08,
    wmax::Union{Nothing,Real}=nothing,
    eta::Real=1e-6,
    output_path::AbstractString="data/ultrasmall_delta_probe_r1_G20.jld2",
)
    G = Float64(Gamma)
    rr = Float64(r)
    wm = isnothing(wmax) ? 8.0 * G : Float64(wmax)
    Opos = collect(Float64.(Omega_values))
    Osigned = signed_values_from_positive(Opos)
    freq_step = 1.0 / (Float64(frequency_step_factor) * G)

    omegas, omega_weights = composite_frequency_grid(G; wmax=wm, dense_step=freq_step)
    qs, q_weights = composite_momentum_grid(
        rr;
        endpoint_window=endpoint_window,
        dense_step=q_dense_step,
        coarse_step=q_coarse_step,
    )

    println("Ultra-small-delta full-Green probe")
    println("r = ", rr, ", Gamma = ", G)
    println("1/Gamma = ", 1.0 / G, ", 1/Gamma^2 = ", 1.0 / G^2)
    println("dense frequency step = ", freq_step, ", Nomega = ", length(omegas))
    println("Nq = ", length(qs))

    line = full_and_mm_omega_lines_for_gamma(
        G,
        omegas,
        omega_weights,
        qs,
        q_weights,
        Osigned;
        r=rr,
        eta=eta,
    )
    rows = estimate_lnC_rows(rr, G, Opos, line)

    println()
    println("Omega    Gamma*Omega  Gamma^2*Omega  lnC_full  lnC_mm  lnC_end  Delta_full")
    for row in rows
        println(
            @sprintf(
                "%8.2e  %11.4f  %13.4f  %9.5f  %8.5f  %8.5f  % .6e",
                row.Omega,
                G * row.Omega,
                G^2 * row.Omega,
                row.lnC_full,
                row.lnC_mm,
                row.lnC_end,
                row.Delta_full,
            ),
        )
    end

    mkpath(dirname(output_path))
    jldsave(
        output_path;
        rows=rows,
        r=rr,
        Gamma=G,
        Omega_values=Opos,
        Omega_values_signed=Osigned,
        frequency_step_factor=Float64(frequency_step_factor),
        dense_frequency_step=freq_step,
        omega_grid=omegas,
        omega_weights=omega_weights,
        q_grid=qs,
        q_weights=q_weights,
        raw=Dict(
            "Pi00_full" => line.Pi00_full,
            "Pi00_mm" => line.Pi00_mm,
            "Pi_full" => line.Pi_full,
            "Pi_mm" => line.Pi_mm,
        ),
    )
    println()
    println("saved ", output_path)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
