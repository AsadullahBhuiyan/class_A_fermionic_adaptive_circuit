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

function component_direct_omega_lines_for_gamma(
    Γ::Float64,
    ωs::Vector{Float64},
    dω::Float64,
    qs::Vector{Float64},
    Ω_values::Vector{Float64};
    r::Float64=1.0,
    η::Float64=1e-4,
)
    nt = maxthreadid()
    nΩ = length(Ω_values)

    partial_base_pp = zeros(ComplexF64, nt)
    partial_base_pm = zeros(ComplexF64, nt)
    partial_base_mp = zeros(ComplexF64, nt)
    partial_base_mm = zeros(ComplexF64, nt)
    partial_pp = zeros(ComplexF64, nt, nΩ)
    partial_pm = zeros(ComplexF64, nt, nΩ)
    partial_mp = zeros(ComplexF64, nt, nΩ)
    partial_mm = zeros(ComplexF64, nt, nΩ)

    @threads for jq in eachindex(qs)
        tid = threadid()
        q = qs[jq]
        base_pp_local = 0.0 + 0.0im
        base_pm_local = 0.0 + 0.0im
        base_mp_local = 0.0 + 0.0im
        base_mm_local = 0.0 + 0.0im

        @inbounds for ω in ωs
            g11, g12, g21, g22 = monitored_retarded_green_entries(ω, q, r, Γ, η)
            gpp, gpm, gmp, gmm = rotate_to_sigma_x_entries(g11, g12, g21, g22)
            base_pp_local += gpp * conj(gpp)
            base_pm_local += gpm * conj(gpm)
            base_mp_local += gmp * conj(gmp)
            base_mm_local += gmm * conj(gmm)

            for idx in eachindex(Ω_values)
                h11, h12, h21, h22 = monitored_retarded_green_entries(ω - Ω_values[idx], q, r, Γ, η)
                hpp, hpm, hmp, hmm = rotate_to_sigma_x_entries(h11, h12, h21, h22)
                partial_pp[tid, idx] += gpp * conj(hpp)
                partial_pm[tid, idx] += gpm * conj(hpm)
                partial_mp[tid, idx] += gmp * conj(hmp)
                partial_mm[tid, idx] += gmm * conj(hmm)
            end
        end

        partial_base_pp[tid] += base_pp_local
        partial_base_pm[tid] += base_pm_local
        partial_base_mp[tid] += base_mp_local
        partial_base_mm[tid] += base_mm_local
    end

    prefactor = dω / (2π * length(qs))
    return (
        Πpp00=prefactor * sum(partial_base_pp),
        Πpm00=prefactor * sum(partial_base_pm),
        Πmp00=prefactor * sum(partial_base_mp),
        Πmm00=prefactor * sum(partial_base_mm),
        Πpp=prefactor .* vec(sum(partial_pp; dims=1)),
        Πpm=prefactor .* vec(sum(partial_pm; dims=1)),
        Πmp=prefactor .* vec(sum(partial_mp; dims=1)),
        Πmm=prefactor .* vec(sum(partial_mm; dims=1)),
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
        x = row.Ωfit
        yfull = row.direct_values ./ (x .^ 2)
        ypp = row.pp_values ./ (x .^ 2)
        ypm = row.pm_values ./ (x .^ 2)
        ymm = row.mm_values ./ (x .^ 2)
        ytotal = row.component_total_values ./ (x .^ 2)

        plt = plot(
            x,
            yfull;
            xscale=:log10,
            label="full direct",
            color=:black,
            linewidth=2.2,
            marker=:circle,
            markersize=4,
            markerstrokewidth=0,
            xlabel="Ω",
            ylabel="Γ[Π(Ω,0)-Π(0,0)] / Ω²",
            title="Γ = $(round(row.Γ; digits=1))",
        )
        plot!(plt, x, ypp; label="exact ++", color=:blue, linewidth=2.0)
        plot!(plt, x, ypm; label="exact +-/ -+", color=:green, linewidth=2.0)
        plot!(plt, x, ymm; label="exact --", color=:red, linewidth=2.0)
        plot!(plt, x, ytotal; label="exact rotated total", color=:orange, linewidth=2.0, linestyle=:dash)
        push!(panels, plt)
    end
    plt = plot(panels...; layout=(length(results), 1), size=(1000, 340 * length(results)))
    savefig(plt, output_pdf)
end

function main(;
    data_dir::AbstractString="data/smallomega_direct_lines_Nw32768_Nq2048_wmax640",
    Γs::AbstractVector=[20.0, 40.0, 80.0],
    Ωfit_max::Float64=0.0015,
    model_Nω::Union{Nothing,Int}=nothing,
    model_Nq::Union{Nothing,Int}=nothing,
    model_ωmax::Union{Nothing,Float64}=nothing,
    r::Float64=1.0,
    η::Float64=1e-4,
    output_path::AbstractString="data/edge_Aomega_estimate.jld2",
    output_plot::AbstractString="note/edge_Aomega_model_comparison.pdf",
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
        Ω_values = Float64.(data["Ω_values"])
        direct_values = real.(Γ .* (data["ΠΩ"] .- data["Π00"]))
        sel = findall(Ω -> Ω <= Ωfit_max, Ω_values)
        Ωfit = Ω_values[sel]
        direct_fit_values = direct_values[sel]

        comp = component_direct_omega_lines_for_gamma(Γ, ωs, dω, qs, Ωfit; r=r, η=η)
        pp_values = real.(Γ .* (comp.Πpp .- comp.Πpp00))
        pm_values = real.(Γ .* (comp.Πpm .- comp.Πpm00))
        mp_values = real.(Γ .* (comp.Πmp .- comp.Πmp00))
        mm_values = real.(Γ .* (comp.Πmm .- comp.Πmm00))
        mixed_values = pm_values .+ mp_values
        component_total_values = pp_values .+ mixed_values .+ mm_values

        fit_direct = fit_quadratic(Ωfit, direct_fit_values)
        fit_pp = fit_quadratic(Ωfit, pp_values)
        fit_pm = fit_quadratic(Ωfit, pm_values)
        fit_mp = fit_quadratic(Ωfit, mp_values)
        fit_mixed = fit_quadratic(Ωfit, mixed_values)
        fit_mm = fit_quadratic(Ωfit, mm_values)
        fit_total = fit_quadratic(Ωfit, component_total_values)

        push!(results, (
            Γ=Γ,
            Ωfit=Ωfit,
            direct_values=direct_fit_values,
            pp_values=pp_values,
            pm_values=pm_values,
            mp_values=mp_values,
            mixed_values=mixed_values,
            mm_values=mm_values,
            component_total_values=component_total_values,
            direct_coeff=fit_direct.coeff,
            pp_coeff=fit_pp.coeff,
            pm_coeff=fit_pm.coeff,
            mp_coeff=fit_mp.coeff,
            mixed_coeff=fit_mixed.coeff,
            mm_coeff=fit_mm.coeff,
            total_coeff=fit_total.coeff,
            direct_rel_rms=fit_direct.rel_rms,
            pp_rel_rms=fit_pp.rel_rms,
            pm_rel_rms=fit_pm.rel_rms,
            mp_rel_rms=fit_mp.rel_rms,
            mixed_rel_rms=fit_mixed.rel_rms,
            mm_rel_rms=fit_mm.rel_rms,
            total_rel_rms=fit_total.rel_rms,
        ))
    end

    plot_ratio_comparison(results, output_plot)
    jldsave(
        output_path;
        data_dir,
        Γs=Float64.(Γs),
        Ωfit_max,
        Nω_model=Nω,
        Nq_model=Nq,
        ωmax_model=ωmax,
        output_plot,
        results,
    )

    println("saved ", output_path)
    println("saved ", output_plot)
    for row in results
        println(
            @sprintf(
                "Γ=%.1f full=%.8f exact_pp=%.8f exact_pm=%.8f exact_mp=%.8f mixed=%.8f exact_mm=%.8f rotated_total=%.8f",
                row.Γ,
                -row.direct_coeff,
                -row.pp_coeff,
                -row.pm_coeff,
                -row.mp_coeff,
                -row.mixed_coeff,
                -row.mm_coeff,
                -row.total_coeff,
            ),
        )
    end
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
