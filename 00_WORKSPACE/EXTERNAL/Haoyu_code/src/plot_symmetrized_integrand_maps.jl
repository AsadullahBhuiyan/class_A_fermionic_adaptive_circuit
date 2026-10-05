using JLD2
using Plots
using Printf
using Statistics

include("taylor_fit_convolution.jl")

@inline function overlap_entries(
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

@inline function kernel_overlap(
    ω::Float64,
    q::Float64,
    Ω::Float64,
    k::Float64,
    r::Float64,
    Γ::Float64,
    η::Float64,
)
    g11, g12, g21, g22 = monitored_retarded_green_entries(ω, q, r, Γ, η)
    h11, h12, h21, h22 = monitored_retarded_green_entries(ω - Ω, q - k, r, Γ, η)
    return overlap_entries(g11, g12, g21, g22, h11, h12, h21, h22)
end

@inline function symmetrized_real_difference_integrand(
    ω::Float64,
    q::Float64,
    Ω::Float64,
    k::Float64,
    r::Float64,
    Γ::Float64,
    η::Float64,
)
    base = kernel_overlap(ω, q, 0.0, 0.0, r, Γ, η)
    plus = kernel_overlap(ω, q, Ω, k, r, Γ, η)
    minus = kernel_overlap(ω, q, -Ω, -k, r, Γ, η)
    return Γ * real(0.5 * (plus + minus) - base)
end

function signed_q_grid(Nq::Int)
    qs = Vector{Float64}(undef, Nq)
    dq = 2π / Nq
    @inbounds for j in 1:Nq
        q = -π + (j - 0.5) * dq
        qs[j] = q
    end
    return qs, dq
end

function midpoint_ω_grid(Nω::Int, ωmax::Float64)
    dω = 2 * ωmax / Nω
    ωs = Vector{Float64}(undef, Nω)
    @inbounds for i in 1:Nω
        ωs[i] = -ωmax + (i - 0.5) * dω
    end
    return ωs, dω
end

function build_integrand_map(
    Γ::Float64,
    Ω::Float64,
    k::Float64;
    Nω::Int=1024,
    Nq::Int=1024,
    ωmax::Float64=4.0,
    r::Float64=1.0,
    η::Float64=1e-4,
)
    ωs, dω = midpoint_ω_grid(Nω, ωmax)
    qs, dq = signed_q_grid(Nq)
    values = Matrix{Float64}(undef, Nω, Nq)

    @inbounds for j in eachindex(qs)
        q = qs[j]
        for i in eachindex(ωs)
            values[i, j] = symmetrized_real_difference_integrand(ωs[i], q, Ω, k, r, Γ, η)
        end
    end

    prefactor = dω * dq / (2π)^2
    return (ωs=ωs, qs=qs, values=values, prefactor=prefactor)
end

function touching_fraction(ωs::Vector{Float64}, qs::Vector{Float64}, values::Matrix{Float64})
    total = sum(abs, values)
    if iszero(total)
        return 0.0
    end

    mask = falses(size(values))
    @inbounds for j in eachindex(qs), i in eachindex(ωs)
        upper = abs(qs[j] - π / 2) <= 0.25 && abs(ωs[i] - 1.0) <= 0.5
        lower = abs(qs[j] + π / 2) <= 0.25 && abs(ωs[i] + 1.0) <= 0.5
        mask[i, j] = upper || lower
    end
    return sum(abs, values[mask]) / total
end

function region_diagnostics(
    ωs::Vector{Float64},
    qs::Vector{Float64},
    values::Matrix{Float64},
    prefactor::Float64,
    Γ::Float64,
)
    total_abs = sum(abs, values)
    total_integral = sum(values) * prefactor
    width = 4.0 / Γ

    edge_mask = falses(size(values))
    touch_mask = falses(size(values))
    @inbounds for j in eachindex(qs), i in eachindex(ωs)
        edge_mask[i, j] = abs(ωs[i] - sin(qs[j])) <= width
        upper = abs(qs[j] - π / 2) <= 0.25 && abs(ωs[i] - 1.0) <= 0.5
        lower = abs(qs[j] + π / 2) <= 0.25 && abs(ωs[i] + 1.0) <= 0.5
        touch_mask[i, j] = upper || lower
    end
    edge_only_mask = edge_mask .& .!touch_mask

    return (
        edge_width=width,
        touching_fraction_abs=iszero(total_abs) ? 0.0 : sum(abs, values[touch_mask]) / total_abs,
        edge_fraction_abs=iszero(total_abs) ? 0.0 : sum(abs, values[edge_mask]) / total_abs,
        edge_only_fraction_abs=iszero(total_abs) ? 0.0 : sum(abs, values[edge_only_mask]) / total_abs,
        total_integral=total_integral,
        edge_integral=sum(values[edge_mask]) * prefactor,
        edge_only_integral=sum(values[edge_only_mask]) * prefactor,
        touching_integral=sum(values[touch_mask]) * prefactor,
        positive_integral=sum(max.(values, 0.0)) * prefactor,
        negative_integral=sum(min.(values, 0.0)) * prefactor,
    )
end

function plot_integrand_figure(
    Γs::Vector{Float64},
    maps::Vector{NamedTuple},
    title_prefix::AbstractString,
    output_pdf::AbstractString,
)
    panels = Plots.Plot[]
    for (Γ, map) in zip(Γs, maps)
        values = map.values
        vmax = quantile(abs.(vec(values)), 0.995)
        full = heatmap(
            map.qs,
            map.ωs,
            values;
            color=:balance,
            clims=(-vmax, vmax),
            xlabel="q",
            ylabel="ω",
            title="$(title_prefix), Γ = $(round(Γ; digits=1))",
            aspect_ratio=:auto,
        )
        vline!(full, [-π/2, π/2]; color=:black, linewidth=1.0, linestyle=:dash, label=nothing)
        hline!(full, [-1.0, 1.0]; color=:black, linewidth=1.0, linestyle=:dash, label=nothing)

        qsel = findall(q -> abs(q - π / 2) <= 0.25, map.qs)
        ωsel = findall(ω -> abs(ω - 1.0) <= 0.5, map.ωs)
        zoom = heatmap(
            map.qs[qsel],
            map.ωs[ωsel],
            values[ωsel, qsel];
            color=:balance,
            clims=(-vmax, vmax),
            xlabel="q",
            ylabel="ω",
            title=@sprintf("upper touching box, fraction = %.1f%%", 100 * touching_fraction(map.ωs, map.qs, values)),
            aspect_ratio=:auto,
        )
        vline!(zoom, [π / 2]; color=:black, linewidth=1.0, linestyle=:dash, label=nothing)
        hline!(zoom, [1.0]; color=:black, linewidth=1.0, linestyle=:dash, label=nothing)

        push!(panels, full)
        push!(panels, zoom)
    end

    plt = plot(panels...; layout=(length(Γs), 2), size=(1200, 320 * length(Γs)))
    savefig(plt, output_pdf)
end

function main(;
    Γs::Vector{Float64}=[20.0, 40.0, 80.0],
    Ω_probe::Float64=0.0015,
    k_probe::Float64=0.0015,
    Nω::Int=1024,
    Nq::Int=1024,
    ωmax::Float64=4.0,
    r::Float64=1.0,
    η::Float64=1e-4,
    output_prefix::AbstractString="note/symmetrized_integrand",
)
    k_maps = NamedTuple[]
    Ω_maps = NamedTuple[]
    rows = NamedTuple[]

    for Γ in Γs
        k_map = build_integrand_map(Γ, 0.0, k_probe; Nω=Nω, Nq=Nq, ωmax=ωmax, r=r, η=η)
        Ω_map = build_integrand_map(Γ, Ω_probe, 0.0; Nω=Nω, Nq=Nq, ωmax=ωmax, r=r, η=η)
        k_diag = region_diagnostics(k_map.ωs, k_map.qs, k_map.values, k_map.prefactor, Γ)
        Ω_diag = region_diagnostics(Ω_map.ωs, Ω_map.qs, Ω_map.values, Ω_map.prefactor, Γ)
        push!(k_maps, k_map)
        push!(Ω_maps, Ω_map)
        push!(rows, (
            Γ=Γ,
            kind="k",
            probe=k_probe,
            touching_fraction=k_diag.touching_fraction_abs,
            edge_fraction=k_diag.edge_fraction_abs,
            edge_only_fraction=k_diag.edge_only_fraction_abs,
            edge_width=k_diag.edge_width,
            total_integral=k_diag.total_integral,
            edge_integral=k_diag.edge_integral,
            edge_only_integral=k_diag.edge_only_integral,
            touching_integral=k_diag.touching_integral,
            positive_integral=k_diag.positive_integral,
            negative_integral=k_diag.negative_integral,
        ))
        push!(rows, (
            Γ=Γ,
            kind="omega",
            probe=Ω_probe,
            touching_fraction=Ω_diag.touching_fraction_abs,
            edge_fraction=Ω_diag.edge_fraction_abs,
            edge_only_fraction=Ω_diag.edge_only_fraction_abs,
            edge_width=Ω_diag.edge_width,
            total_integral=Ω_diag.total_integral,
            edge_integral=Ω_diag.edge_integral,
            edge_only_integral=Ω_diag.edge_only_integral,
            touching_integral=Ω_diag.touching_integral,
            positive_integral=Ω_diag.positive_integral,
            negative_integral=Ω_diag.negative_integral,
        ))
    end

    plot_integrand_figure(Γs, k_maps, @sprintf("Γ Re[Π(0,k)-Π(0,0)] integrand, k = %.4g", k_probe), "$(output_prefix)_k.pdf")
    plot_integrand_figure(Γs, Ω_maps, @sprintf("Γ Re[Π(Ω,0)-Π(0,0)] integrand, Ω = %.4g", Ω_probe), "$(output_prefix)_omega.pdf")

    jldsave(
        "$(output_prefix)_summary.jld2";
        Γs,
        Ω_probe,
        k_probe,
        Nω,
        Nq,
        ωmax,
        rows,
        output_k_pdf="$(output_prefix)_k.pdf",
        output_omega_pdf="$(output_prefix)_omega.pdf",
    )

    println("saved $(output_prefix)_k.pdf")
    println("saved $(output_prefix)_omega.pdf")
    for row in rows
        println(
            @sprintf(
                "%s Γ=%.1f touch_abs=%.4f edge_abs=%.4f total=% .6e edge=% .6e touch=% .6e",
                row.kind,
                row.Γ,
                row.touching_fraction,
                row.edge_fraction,
                row.total_integral,
                row.edge_integral,
                row.touching_integral,
            ),
        )
    end
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
