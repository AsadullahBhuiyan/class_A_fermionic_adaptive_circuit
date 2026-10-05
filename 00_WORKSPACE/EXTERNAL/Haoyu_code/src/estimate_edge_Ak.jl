using JLD2
using Plots
using Printf
using Statistics
using Base.Threads: @threads, maxthreadid, threadid

include("taylor_fit_convolution.jl")

function gamma_token(Γ::Real)
    return replace(@sprintf("%.3f", float(Γ)), "." => "p")
end

function midpoint_frequency_grid(Nω::Int, ωmax::Float64)
    dω = 2 * ωmax / Nω
    ωs = Vector{Float64}(undef, Nω)
    @inbounds for i in 1:Nω
        ωs[i] = -ωmax + (i - 0.5) * dω
    end
    return ωs, dω
end

function uniform_momentum_grid(Nq::Int)
    dq = 2π / Nq
    qs = Vector{Float64}(undef, Nq)
    @inbounds for j in 1:Nq
        qs[j] = (j - 1) * dq
    end
    return qs, dq
end

@inline edge_weight(q::Float64) = max(cos(q) * (2 - cos(q)), 0.0)
@inline edge_energy(q::Float64) = sin(q)
@inline edge_width(q::Float64, Γ::Float64) = 0.5 * Γ * edge_weight(q)

@inline function edge_overlap_kernel(q::Float64, k::Float64, Γ::Float64)
    z1 = edge_weight(q)
    z2 = edge_weight(q - k)
    if z1 == 0.0 || z2 == 0.0
        return 0.0
    end

    e1 = edge_energy(q)
    e2 = edge_energy(q - k)
    g1 = edge_width(q, Γ)
    g2 = edge_width(q - k, Γ)
    return Γ * z1 * z2 * (g1 + g2) / ((e1 - e2)^2 + (g1 + g2)^2)
end

@inline function edge_overlap_kernel_large_gamma(q::Float64, k::Float64)
    z1 = edge_weight(q)
    z2 = edge_weight(q - k)
    if z1 == 0.0 || z2 == 0.0
        return 0.0
    end
    return 2 * z1 * z2 / (z1 + z2)
end

function integrate_periodic(f, Nq::Int=200_000)
    dq = 2π / Nq
    acc = 0.0
    @inbounds for j in 1:Nq
        q = -π + (j - 0.5) * dq
        acc += f(q)
    end
    return dq * acc / (2π)
end

function edge_delta_pi(k::Float64, Γ::Float64; Nq::Int=200_000)
    return integrate_periodic(Nq) do q
        0.5 * (edge_overlap_kernel(q, k, Γ) + edge_overlap_kernel(q, -k, Γ)) - edge_overlap_kernel(q, 0.0, Γ)
    end
end

function edge_delta_pi_large_gamma(k::Float64; Nq::Int=200_000)
    return integrate_periodic(Nq) do q
        0.5 * (edge_overlap_kernel_large_gamma(q, k) + edge_overlap_kernel_large_gamma(q, -k)) - edge_overlap_kernel_large_gamma(q, 0.0)
    end
end

@inline function rotate_to_sigma_x_entries(
    g11::ComplexF64,
    g12::ComplexF64,
    g21::ComplexF64,
    g22::ComplexF64,
)
    gpp = 0.5 * (g11 + g12 + g21 + g22)
    gpm = 0.5 * (g11 - g12 + g21 - g22)
    gmp = 0.5 * (g11 + g12 - g21 - g22)
    gmm = 0.5 * (g11 - g12 - g21 + g22)
    return gpp, gpm, gmp, gmm
end

function component_direct_lines_for_gamma(
    Γ::Float64,
    ωs::Vector{Float64},
    dω::Float64,
    qs::Vector{Float64},
    k_values::Vector{Float64};
    r::Float64=1.0,
    η::Float64=1e-4,
)
    nt = maxthreadid()
    nk = length(k_values)

    partial_base_pp = zeros(ComplexF64, nt)
    partial_base_mixed = zeros(ComplexF64, nt)
    partial_base_mm = zeros(ComplexF64, nt)
    partial_pp = zeros(ComplexF64, nt, nk)
    partial_mixed = zeros(ComplexF64, nt, nk)
    partial_mm = zeros(ComplexF64, nt, nk)

    @threads for jq in eachindex(qs)
        tid = threadid()
        q = qs[jq]
        base_pp_local = 0.0 + 0.0im
        base_mixed_local = 0.0 + 0.0im
        base_mm_local = 0.0 + 0.0im

        @inbounds for ω in ωs
            g11, g12, g21, g22 = monitored_retarded_green_entries(ω, q, r, Γ, η)
            gpp, gpm, gmp, gmm = rotate_to_sigma_x_entries(g11, g12, g21, g22)
            base_pp_local += gpp * conj(gpp)
            base_mixed_local += gpm * conj(gpm) + gmp * conj(gmp)
            base_mm_local += gmm * conj(gmm)

            for idx in eachindex(k_values)
                h11, h12, h21, h22 = monitored_retarded_green_entries(ω, q - k_values[idx], r, Γ, η)
                hpp, hpm, hmp, hmm = rotate_to_sigma_x_entries(h11, h12, h21, h22)
                partial_pp[tid, idx] += gpp * conj(hpp)
                partial_mixed[tid, idx] += gpm * conj(hpm) + gmp * conj(hmp)
                partial_mm[tid, idx] += gmm * conj(hmm)
            end
        end

        partial_base_pp[tid] += base_pp_local
        partial_base_mixed[tid] += base_mixed_local
        partial_base_mm[tid] += base_mm_local
    end

    prefactor = dω / (2π * length(qs))
    Πpp00 = prefactor * sum(partial_base_pp)
    Πmixed00 = prefactor * sum(partial_base_mixed)
    Πmm00 = prefactor * sum(partial_base_mm)
    Πpp = prefactor .* vec(sum(partial_pp; dims=1))
    Πmixed = prefactor .* vec(sum(partial_mixed; dims=1))
    Πmm = prefactor .* vec(sum(partial_mm; dims=1))
    return (
        Πpp00=Πpp00,
        Πmixed00=Πmixed00,
        Πmm00=Πmm00,
        Πpp=Πpp,
        Πmixed=Πmixed,
        Πmm=Πmm,
    )
end

function fit_quadratic(x::Vector{Float64}, y::Vector{Float64})
    coeff = sum((x .^ 2) .* y) / sum(x .^ 4)
    fitted = coeff .* (x .^ 2)
    rel_rms = sqrt(mean(abs2, y .- fitted)) / max(maximum(abs.(y)), eps())
    return (coeff=coeff, fitted=fitted, rel_rms=rel_rms)
end

function plot_ratio_comparison(results::AbstractVector, output_pdf::AbstractString)
    panels = Plots.Plot[]
    for row in results
        x = row.kfit
        yfull = row.direct_values ./ (x .^ 2)
        ypp = row.pp_values ./ (x .^ 2)
        ymm = row.mm_values ./ (x .^ 2)
        ytotal = row.component_total_values ./ (x .^ 2)
        ypole = row.edge_values ./ (x .^ 2)

        plt_full = plot(
            x,
            yfull;
            xscale=:log10,
            label="full direct",
            color=:black,
            linewidth=2.2,
            marker=:circle,
            markersize=4,
            markerstrokewidth=0,
            xlabel="k",
            ylabel="Γ[Π(0,k)-Π(0,0)] / k²",
            title="Γ = $(round(row.Γ; digits=1)), full scale",
        )
        plot!(plt_full, x, ypole; label="edge pole", color=:red, linewidth=2.0)
        plot!(plt_full, x, ypp; label="exact ++", color=:blue, linewidth=2.0)
        plot!(plt_full, x, ymm; label="exact --", color=:green, linewidth=2.0)
        plot!(plt_full, x, ytotal; label="exact rotated total", color=:orange, linewidth=2.0, linestyle=:dash)

        ymin = min(minimum(yfull), minimum(ypp), minimum(ymm), minimum(ytotal)) - 0.01
        ymax = max(maximum(yfull), maximum(ypp), maximum(ymm), maximum(ytotal)) + 0.01
        plt_zoom = plot(
            x,
            yfull;
            xscale=:log10,
            label="full direct",
            color=:black,
            linewidth=2.2,
            marker=:circle,
            markersize=4,
            markerstrokewidth=0,
            xlabel="k",
            ylabel="Γ[Π(0,k)-Π(0,0)] / k²",
            ylims=(ymin, ymax),
            title="Γ = $(round(row.Γ; digits=1)), zoom",
        )
        plot!(plt_zoom, x, ypp; label="exact ++", color=:blue, linewidth=2.0)
        plot!(plt_zoom, x, ymm; label="exact --", color=:green, linewidth=2.0)
        plot!(plt_zoom, x, ytotal; label="exact rotated total", color=:orange, linewidth=2.0, linestyle=:dash)

        push!(panels, plt_full)
        push!(panels, plt_zoom)
    end
    plt = plot(panels...; layout=(length(results), 2), size=(1300, 330 * length(results)))
    savefig(plt, output_pdf)
end

function main(;
    data_dir::AbstractString="data/smallk_direct_lines_Nw16384_Nq4096_wmax320",
    Γs::AbstractVector=[20.0, 40.0, 80.0],
    kfit_max::Float64=0.0025,
    Nq_edge_integral::Int=200_000,
    model_Nω::Union{Nothing,Int}=nothing,
    model_Nq::Union{Nothing,Int}=nothing,
    model_ωmax::Union{Nothing,Float64}=nothing,
    r::Float64=1.0,
    η::Float64=1e-4,
    output_path::AbstractString="data/edge_Ak_estimate.jld2",
    output_plot::AbstractString="note/edge_Ak_model_comparison.pdf",
)
    meta = load(joinpath(data_dir, "metadata.jld2"))
    Nω = isnothing(model_Nω) ? Int(meta["Nω"]) : model_Nω
    Nq = isnothing(model_Nq) ? Int(meta["Nq"]) : model_Nq
    ωmax = isnothing(model_ωmax) ? Float64(meta["ωmax"]) : model_ωmax
    ωs, dω = midpoint_frequency_grid(Nω, ωmax)
    qs, _ = uniform_momentum_grid(Nq)

    results = NamedTuple[]
    for Γraw in Γs
        Γ = Float64(Γraw)
        data = load(joinpath(data_dir, "gamma_$(gamma_token(Γ)).jld2"))
        k_values = Float64.(data["k_values"])
        direct_values = real.(Γ .* (data["Πk"] .- data["Π00"]))
        sel = findall(k -> k <= kfit_max, k_values)
        kfit = k_values[sel]
        direct_fit_values = direct_values[sel]

        edge_values = [edge_delta_pi(k, Γ; Nq=Nq_edge_integral) for k in kfit]
        edge_large_gamma_values = [edge_delta_pi_large_gamma(k; Nq=Nq_edge_integral) for k in kfit]

        comp = component_direct_lines_for_gamma(Γ, ωs, dω, qs, kfit; r=r, η=η)
        pp_values = real.(Γ .* (comp.Πpp .- comp.Πpp00))
        mixed_values = real.(Γ .* (comp.Πmixed .- comp.Πmixed00))
        mm_values = real.(Γ .* (comp.Πmm .- comp.Πmm00))
        pp_plus_mixed_values = pp_values .+ mixed_values
        component_total_values = pp_values .+ mixed_values .+ mm_values

        fit_direct = fit_quadratic(kfit, direct_fit_values)
        fit_edge = fit_quadratic(kfit, edge_values)
        fit_edge_inf = fit_quadratic(kfit, edge_large_gamma_values)
        fit_pp = fit_quadratic(kfit, pp_values)
        fit_mixed = fit_quadratic(kfit, mixed_values)
        fit_mm = fit_quadratic(kfit, mm_values)
        fit_pp_plus_mixed = fit_quadratic(kfit, pp_plus_mixed_values)
        fit_total = fit_quadratic(kfit, component_total_values)

        push!(results, (
            Γ=Γ,
            kfit=kfit,
            direct_values=direct_fit_values,
            edge_values=edge_values,
            edge_large_gamma_values=edge_large_gamma_values,
            pp_values=pp_values,
            mixed_values=mixed_values,
            mm_values=mm_values,
            pp_plus_mixed_values=pp_plus_mixed_values,
            component_total_values=component_total_values,
            direct_coeff=fit_direct.coeff,
            edge_coeff=fit_edge.coeff,
            edge_large_gamma_coeff=fit_edge_inf.coeff,
            pp_coeff=fit_pp.coeff,
            mixed_coeff=fit_mixed.coeff,
            mm_coeff=fit_mm.coeff,
            pp_plus_mixed_coeff=fit_pp_plus_mixed.coeff,
            total_coeff=fit_total.coeff,
            direct_rel_rms=fit_direct.rel_rms,
            edge_rel_rms=fit_edge.rel_rms,
            edge_large_gamma_rel_rms=fit_edge_inf.rel_rms,
            pp_rel_rms=fit_pp.rel_rms,
            mixed_rel_rms=fit_mixed.rel_rms,
            mm_rel_rms=fit_mm.rel_rms,
            pp_plus_mixed_rel_rms=fit_pp_plus_mixed.rel_rms,
            total_rel_rms=fit_total.rel_rms,
        ))
    end

    plot_ratio_comparison(results, output_plot)
    jldsave(
        output_path;
        data_dir,
        Γs=Float64.(Γs),
        kfit_max,
        Nω_model=Nω,
        Nq_model=Nq,
        ωmax_model=ωmax,
        Nq_edge_integral,
        output_plot,
        results,
    )

    println("saved ", output_path)
    println("saved ", output_plot)
    for row in results
        println(
            @sprintf(
                "Γ=%.1f full=%.8f pole=%.8f exact_pp=%.8f mixed=%.8f mm=%.8f pp+mixed=%.8f rotated_total=%.8f",
                row.Γ,
                -row.direct_coeff,
                -row.edge_coeff,
                -row.pp_coeff,
                -row.mixed_coeff,
                -row.mm_coeff,
                -row.pp_plus_mixed_coeff,
                -row.total_coeff,
            ),
        )
    end
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
