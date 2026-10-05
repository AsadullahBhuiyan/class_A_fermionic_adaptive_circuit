using JLD2
using Printf

include("collect_cubic_triangle_data.jl")

function gamma_token(Γ::Real)
    return replace(@sprintf("%.3f", float(Γ)), "." => "p")
end

@inline edge_weight(q::Float64) = max(cos(q) * (2 - cos(q)), 0.0)

@inline function ghat_pp_signstrip(q::Float64, Γ::Float64)
    return 2 / (im - 2 * sin(q) / Γ)
end

@inline function ghat_mm_signstrip(q::Float64, Γ::Float64)
    Zq = edge_weight(q)
    sq = sin(q)
    return 2 * Γ * sq / (im * Γ * sq + 2 * Zq)
end

@inline function cycle_chat(ghat::ComplexF64)
    Δhat = ghat - conj(ghat)
    return -(ghat * conj(ghat) * Δhat + Δhat^3)
end

function integrate_edge(f; Nq::Int=200_000)
    dq = π / Nq
    acc = 0.0 + 0.0im
    @inbounds for jq in 1:Nq
        q = -π / 2 + (jq - 0.5) * dq
        acc += f(q)
    end
    return dq * acc / (2π)^2
end

function estimate_diagonal_cycles_for_gamma(Γ::Float64; Nq::Int=200_000)
    analytic_pp = integrate_edge(q -> cycle_chat(ghat_pp_signstrip(q, Γ)); Nq=Nq)
    analytic_mm = integrate_edge(q -> cycle_chat(ghat_mm_signstrip(q, Γ)); Nq=Nq)
    coeff_inf = -(24 / π) * im
    coeff_one_cycle_inf = -(12 / π) * im
    return (
        Γ=Γ,
        analytic_pp=analytic_pp,
        analytic_mm=analytic_mm,
        analytic_sum=analytic_pp + analytic_mm,
        coeff_one_cycle_inf=coeff_one_cycle_inf,
        coeff_sum_inf=coeff_inf,
    )
end

function load_measured_diagonal_cycles(breakdown_dir::AbstractString, Γ::Float64)
    data = load(joinpath(breakdown_dir, "gamma_$(gamma_token(Γ)).jld2"))
    cycle_labels = Vector{String}(data["cycle_labels"])
    cycle_sum_dΩ1 = Vector{ComplexF64}(data["cycle_sum_dΩ1"])
    idx_pp = findfirst(==("+++"), cycle_labels)
    idx_mm = findfirst(==("---"), cycle_labels)
    @assert !isnothing(idx_pp) && !isnothing(idx_mm)
    return (
        measured_pp=cycle_sum_dΩ1[idx_pp],
        measured_mm=cycle_sum_dΩ1[idx_mm],
        measured_sum=cycle_sum_dΩ1[idx_pp] + cycle_sum_dΩ1[idx_mm],
        measured_total=ComplexF64(data["total_dΩ1"]),
    )
end

function estimate_cubic_diagonal_cycles(;
    Γs::AbstractVector=[20.0, 40.0, 80.0],
    Nq::Int=200_000,
    breakdown_dir::AbstractString="data/cubic_linear_omega_xbasis_sign_Nw12800_Nk256_dw0p050",
    output_path::AbstractString="data/cubic_diagonal_cycle_estimates.jld2",
)
    rows = NamedTuple[]
    for Γraw in Γs
        Γ = Float64(Γraw)
        analytic = estimate_diagonal_cycles_for_gamma(Γ; Nq=Nq)
        measured = load_measured_diagonal_cycles(breakdown_dir, Γ)
        push!(rows, merge(analytic, measured))
    end

    jldsave(output_path; Γs=collect(Float64.(Γs)), Nq, breakdown_dir, rows)

    println("saved ", output_path)
    for row in rows
        println("Γ = ", row.Γ)
        println("  analytic +++ = ", row.analytic_pp)
        println("  measured +++ = ", row.measured_pp)
        println("  analytic --- = ", row.analytic_mm)
        println("  measured --- = ", row.measured_mm)
        println("  analytic diag sum = ", row.analytic_sum)
        println("  measured diag sum = ", row.measured_sum)
        println("  measured total = ", row.measured_total)
        println("  one-cycle asymptotic = ", row.coeff_one_cycle_inf)
        println("  diagonal asymptotic sum = ", row.coeff_sum_inf)
    end

    return rows
end

main(; kwargs...) = estimate_cubic_diagonal_cycles(; kwargs...)

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
